import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly


CLK_PERIOD_NS = 10
RESET_CYCLES = 5


async def reset_dut(dut):
    dut.rst.value = 1
    dut.data_in.value = 0
    dut.in_valid.value = 0
    dut.out_ready.value = 0

    for _ in range(RESET_CYCLES):
        await RisingEdge(dut.clk)

    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def send_one(dut, value: int):
    dut.data_in.value = value
    dut.in_valid.value = 1

    while True:
        await ReadOnly()
        if int(dut.in_ready.value) == 1:
            break
        await RisingEdge(dut.clk)

    await RisingEdge(dut.clk)

    dut.in_valid.value = 0
    dut.data_in.value = 0


async def send_batch(dut, values):
    for v in values:
        await send_one(dut, v)


async def recv_one(dut) -> int:
    dut.out_ready.value = 1

    while True:
        await ReadOnly()
        if int(dut.out_valid.value) == 1:
            value = int(dut.data_out.value)
            await RisingEdge(dut.clk)  # consume handshake
            dut.out_ready.value = 0
            return value
        await RisingEdge(dut.clk)


@cocotb.test()
async def test_qsum_basic(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    await reset_dut(dut)

    batch = [1, 2, 3, 4]
    expected = sum(batch)

    await send_batch(dut, batch)
    result = await recv_one(dut)

    dut._log.info(f"batch={batch}, result={result}")
    assert result == expected, f"Expected {expected}, got {result}"


@cocotb.test()
async def test_qsum_two_batches_and_reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    await reset_dut(dut)

    batch1 = [1, 2, 3, 4]
    batch2 = [5, 6, 7, 8]

    expected1 = sum(batch1)
    expected2 = sum(batch2)

    result_before = int(dut.data_out.value)
    assert result_before == 0, f"Expected reset output 0, got {result_before}"

    await send_batch(dut, batch1)
    result1 = await recv_one(dut)
    assert result1 == expected1, f"Batch1: expected {expected1}, got {result1}"
    dut._log.info(f"batch1={batch1}, result={result1}")
    await RisingEdge(dut.clk)
    assert int(dut.out_valid.value) == 0, "out_valid should deassert after consume"
    assert int(dut.data_out.value) == 0, "data_out should reset to 0 after consume"
    assert int(dut.in_ready.value) == 1, "in_ready should go high again after consume"
    
    await send_batch(dut, batch2)
    result2 = await recv_one(dut)
    dut._log.info(f"batch2={batch2}, result={result2}")

    assert result2 == expected2, f"Batch2: expected {expected2}, got {result2}"
    await RisingEdge(dut.clk)
    assert int(dut.out_valid.value) == 0, "out_valid should deassert after consume"
    assert int(dut.data_out.value) == 0, "data_out should reset to 0 after consume"
    assert int(dut.in_ready.value) == 1, "in_ready should go high again after consume"

