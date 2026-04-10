# run_synth.tcl
#
# Usage:
#   vivado -mode batch -source run_synth.tcl -tclargs <top> <part> <rtl_dir> <out_dir> [clk_port] [clk_period_ns]

proc extract_site_used {report_text site_name} {
    foreach line [split $report_text "\n"] {
        set trimmed [string trim $line]

        if {![string match "|*" $trimmed]} {
            continue
        }

        set parts [split $trimmed "|"]
        set cols {}
        foreach p $parts {
            set t [string trim $p]
            if {$t ne ""} {
                lappend cols $t
            }
        }

        if {[llength $cols] >= 2} {
            set row_name [lindex $cols 0]
            set used_val [lindex $cols 1]

            if {$row_name eq $site_name} {
                regsub -all {,} $used_val {} used_val
                return $used_val
            }
        }
    }

    return 0
}

proc json_num_or_null {value} {
    if {$value eq ""} {
        return "null"
    }
    return $value
}

proc json_string_or_null {value} {
    if {$value eq ""} {
        return "null"
    }
    set escaped $value
    regsub -all {\\} $escaped {\\\\} escaped
    regsub -all {"} $escaped {\\"} escaped
    return "\"$escaped\""
}

proc get_wns {} {
    set paths [get_timing_paths -setup -nworst 1]
    if {[llength $paths] == 0} {
        return ""
    }
    return [get_property SLACK [lindex $paths 0]]
}

proc get_whs {} {
    set paths [get_timing_paths -hold -nworst 1]
    if {[llength $paths] == 0} {
        return ""
    }
    return [get_property SLACK [lindex $paths 0]]
}

proc write_summary_json {out_path top_name part_name clk_port clk_period} {
    set util_rpt [report_utilization -return_string]

    # Utilization extraction
    set lut [extract_site_used $util_rpt "CLB LUTs*"]
    if {$lut == 0} {
        set lut [extract_site_used $util_rpt "CLB LUTs"]
    }

    set ff [extract_site_used $util_rpt "CLB Registers"]

    set bram36 [extract_site_used $util_rpt "RAMB36/FIFO*"]
    if {$bram36 == 0} {
        set bram36 [extract_site_used $util_rpt "RAMB36"]
    }

    set bram18 [extract_site_used $util_rpt "RAMB18"]
    set uram [extract_site_used $util_rpt "URAM"]

    set dsp [extract_site_used $util_rpt "DSPs"]
    if {$dsp == 0} {
        set dsp [extract_site_used $util_rpt "DSP Blocks"]
    }

    # Timing extraction via timing path objects, not report text parsing
    set wns [get_wns]
    set whs [get_whs]

    # Leave these blank for now unless you decide to compute them another way
    set tns ""
    set ths ""

    # Derived Fmax estimate from setup slack
    set achieved_period_ns ""
    set fmax_mhz ""

    if {$clk_period ne "" && $wns ne ""} {
        set achieved_period_ns [expr {double($clk_period) - double($wns)}]
        if {$achieved_period_ns > 0.0} {
            set fmax_mhz [expr {1000.0 / $achieved_period_ns}]
        }
    }

    set fp [open $out_path "w"]

    puts $fp "{"
    puts $fp "  \"part\": [json_string_or_null $part_name],"
    puts $fp "  \"top\": [json_string_or_null $top_name],"
    puts $fp "  \"clock\": {"
    puts $fp "    \"port\": [json_string_or_null $clk_port],"
    puts $fp "    \"target_period_ns\": [json_num_or_null $clk_period]"
    puts $fp "  },"
    puts $fp "  \"utilization\": {"
    puts $fp "    \"lut\": $lut,"
    puts $fp "    \"ff\": $ff,"
    puts $fp "    \"bram36\": $bram36,"
    puts $fp "    \"bram18\": $bram18,"
    puts $fp "    \"uram\": $uram,"
    puts $fp "    \"dsp\": $dsp"
    puts $fp "  },"
    puts $fp "  \"timing\": {"
    puts $fp "    \"wns_ns\": [json_num_or_null $wns],"
    puts $fp "    \"tns_ns\": [json_num_or_null $tns],"
    puts $fp "    \"whs_ns\": [json_num_or_null $whs],"
    puts $fp "    \"ths_ns\": [json_num_or_null $ths],"
    puts $fp "    \"achieved_period_ns\": [json_num_or_null $achieved_period_ns],"
    puts $fp "    \"estimated_fmax_mhz\": [json_num_or_null $fmax_mhz]"
    puts $fp "  }"
    puts $fp "}"

    close $fp
}

set top_name  [lindex $argv 0]
set part_name [lindex $argv 1]
set rtl_dir   [file normalize [lindex $argv 2]]
set out_dir   [file normalize [lindex $argv 3]]

if {$top_name eq "" || $part_name eq "" || $rtl_dir eq "" || $out_dir eq ""} {
    puts "ERROR: Missing arguments."
    puts "Usage: vivado -mode batch -source run_synth.tcl -tclargs <top> <part> <rtl_dir> <out_dir> [clk_port] [clk_period_ns]"
    exit 1
}

file mkdir $out_dir

puts "Top module : $top_name"
puts "Part       : $part_name"
puts "RTL dir    : $rtl_dir"
puts "Output dir : $out_dir"

# Top-level files in rtl_dir
set top_files   [glob -nocomplain -directory $rtl_dir *.sv]
set top_vfiles  [glob -nocomplain -directory $rtl_dir *.v]

# Files directly under src/
set src_dir [file join $rtl_dir src]
set src_files {}
set src_svfiles {}
if {[file exists $src_dir] && [file isdirectory $src_dir]} {
    set src_files   [glob -nocomplain -directory $src_dir *.v]
    set src_svfiles [glob -nocomplain -directory $src_dir *.sv]
}

# Files under src/static/
set static_dir [file join $src_dir static]
set static_files {}
set static_svfiles {}
if {[file exists $static_dir] && [file isdirectory $static_dir]} {
    set static_files   [glob -nocomplain -directory $static_dir *.v]
    set static_svfiles [glob -nocomplain -directory $static_dir *.sv]
}

# Combine all RTL files
set rtl_files [concat $top_files $top_vfiles $src_files $src_svfiles $static_files $static_svfiles]

if {[llength $rtl_files] == 0} {
    puts "ERROR: No RTL files found."
    exit 1
}

puts "RTL files to read:"
foreach f $rtl_files {
    puts "  $f"
}

# Read RTL
foreach f $rtl_files {
    if {[string match *.sv $f]} {
        read_verilog -sv $f
    } else {
        read_verilog $f
    }
}

# Include dirs for any `include usage
if {[file exists $src_dir] && [file isdirectory $src_dir]} {
    set include_dirs [list $rtl_dir $src_dir]
    if {[file exists $static_dir] && [file isdirectory $static_dir]} {
        lappend include_dirs $static_dir
    }
    set_property include_dirs $include_dirs [current_fileset]

    puts "Include dirs:"
    foreach d $include_dirs {
        puts "  $d"
    }
}

# Optional clock args
set clk_port   [lindex $argv 4]
set clk_period [lindex $argv 5]

if {$clk_port ne "" && $clk_period ne ""} {
    puts "Preparing clock constraint: port=$clk_port period=${clk_period}ns"

    set auto_xdc [file join $out_dir auto_clock.xdc]
    set fp [open $auto_xdc "w"]
    puts $fp "create_clock -name $clk_port -period $clk_period \[get_ports $clk_port\]"
    close $fp

    read_xdc $auto_xdc
} else {
    puts "No clock constraint provided."
}

# Run synthesis
synth_design -top $top_name -part $part_name -flatten_hierarchy none

# Reports
check_timing -file [file join $out_dir check_timing.rpt]
report_timing_summary -report_unconstrained -file [file join $out_dir timing_summary.rpt]
report_utilization -file [file join $out_dir utilization.rpt]
report_utilization -hierarchical -file [file join $out_dir utilization_hier.rpt]

# Structured summary
write_summary_json [file join $out_dir summary.json] $top_name $part_name $clk_port $clk_period

# Optional native utilization JSON
catch {
    report_utilization -json -file [file join $out_dir utilization.json]
}

# Save checkpoint
write_checkpoint -force [file join $out_dir post_synth.dcp]

exit