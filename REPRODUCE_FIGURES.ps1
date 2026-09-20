$kgrRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $kgrRoot
try {
    python experiments/plot_formal_v3_results.py
    if ($LASTEXITCODE -ne 0) { throw "plot_formal_v3_results.py failed" }

    python experiments/plot_information_privacy_tradeoff.py
    if ($LASTEXITCODE -ne 0) { throw "plot_information_privacy_tradeoff.py failed" }

    python experiments/plot_relative_mse_improvement.py
    if ($LASTEXITCODE -ne 0) { throw "plot_relative_mse_improvement.py failed" }

    python experiments/plot_kgr_ldp_framework.py
    if ($LASTEXITCODE -ne 0) { throw "plot_kgr_ldp_framework.py failed" }

    python experiments/plot_schema_expansion.py
    if ($LASTEXITCODE -ne 0) { throw "plot_schema_expansion.py failed" }

    python experiments/plot_supplementary_figure_s1.py
    if ($LASTEXITCODE -ne 0) { throw "plot_supplementary_figure_s1.py failed" }

    python experiments/plot_supplementary_figure_s2.py
    if ($LASTEXITCODE -ne 0) { throw "plot_supplementary_figure_s2.py failed" }
}
finally {
    Pop-Location
}
