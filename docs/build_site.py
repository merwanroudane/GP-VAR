"""
Build the static documentation site (GitHub Pages, served from /docs).

    python docs/build_site.py

It converts README.md, docs/USER_GUIDE.md and docs/METHODOLOGY.md to HTML,
assembles a results page from every figure and table in outputs/, copies the
PNG figures into docs/figures and writes docs/*.html with a light,
print-like theme.  Requires the `markdown` package (pip install markdown).
"""
from __future__ import annotations

import glob
import os
import re
import shutil

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
FIG_SRC = os.path.join(ROOT, "outputs", "figures")
TAB_SRC = os.path.join(ROOT, "outputs", "tables")
FIG_DST = os.path.join(DOCS, "figures")
RAW = "https://raw.githubusercontent.com/merwanroudane/GP-VAR/main/"

MD_EXT = ["tables", "fenced_code", "toc", "attr_list", "md_in_html", "pymdownx.arithmatex"]
MD_CFG = {"pymdownx.arithmatex": {"generic": True}}

NAV = [("index.html", "Overview"), ("concepts.html", "Concepts"), ("results.html", "Results"),
       ("guide.html", "User guide"), ("methodology.html", "Methodology"),
       ("https://github.com/merwanroudane/GP-VAR", "GitHub"), ("https://pypi.org/project/gpvar/", "PyPI")]


def md(text: str) -> str:
    html = markdown.markdown(text, extensions=MD_EXT, extension_configs=MD_CFG)
    return html.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")


def page(title: str, body: str, active: str, subtitle: str = "") -> str:
    nav = "".join(
        f'<a href="{href}"{" class=active" if href == active else ""}>{label}</a>' for href, label in NAV)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · gpvar</title>
<link rel="stylesheet" href="style.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@400;600&family=Source+Code+Pro:wght@400;500&display=swap" rel="stylesheet">
<script>
MathJax = {{ tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }} }};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
<header class="masthead">
  <div class="wrap">
    <div class="brand"><a href="index.html">gpvar</a><span class="tagline">Gaussian Process Vector Autoregressions with Stochastic Volatility in Python</span></div>
    <nav>{nav}</nav>
  </div>
</header>
<main class="wrap">
<h1 class="pagetitle">{title}</h1>
{f'<p class="subtitle">{subtitle}</p>' if subtitle else ''}
{body}
</main>
<footer class="wrap">
  <p>gpvar · Dr Merwan Roudane · <a href="mailto:merwanroudane920@gmail.com">merwanroudane920@gmail.com</a> · <a href="https://github.com/merwanroudane/GP-VAR">github.com/merwanroudane/GP-VAR</a> · MIT License.<br>
  Implements Hauzenberger, Huber, Marcellino and Petz (2022), <em>Gaussian Process Vector Autoregressions and Macroeconomic Uncertainty</em>, <a href="https://arxiv.org/abs/2112.01995">arXiv:2112.01995</a>. Data: FRED-QD (Federal Reserve Bank of St. Louis) and Jurado, Ludvigson and Ng (2015).</p>
</footer>
</body>
</html>
"""


def table_html(name: str) -> str:
    path = os.path.join(TAB_SRC, name + ".md")
    if not os.path.exists(path):
        return ""
    return md(open(path, encoding="utf-8").read())


def figure(name: str, caption: str) -> str:
    return (f'<figure><a href="figures/{name}.png"><img src="figures/{name}.png" alt="{caption}"></a>'
            f'<figcaption>{caption} <span class="dl">(<a href="{RAW}outputs/figures/{name}.pdf">PDF</a>)</span></figcaption></figure>')


# --------------------------------------------------------------------------- #
def build():
    os.makedirs(FIG_DST, exist_ok=True)
    for f in glob.glob(os.path.join(FIG_SRC, "*.png")):
        shutil.copy2(f, FIG_DST)
    open(os.path.join(DOCS, ".nojekyll"), "w").close()

    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    readme = readme.replace(RAW + "outputs/figures/", "figures/")

    # ---------------- index: overview = README sections 1-3 + result highlights + 9-14
    def section(num: int) -> str:
        m = re.search(rf"^## {num}\. .*?(?=^## \d+\. |\Z)", readme, flags=re.M | re.S)
        return m.group(0) if m else ""

    intro = readme.split("## Contents")[0]
    intro = re.sub(r"^# .*\n", "", intro).strip()
    overview = md(intro) + '<div class="grid3">' + "".join(
        f'<a class="card" href="{h}"><h3>{t}</h3><p>{d}</p></a>' for h, t, d in [
            ("concepts.html", "Concepts", "The model, the SV-scaled Gaussian-process priors, the hyperparameter grid, the sampler and generalized impulse responses, explained with the notation of the paper."),
            ("results.html", "Results on real data", "The complete GP-VAR-8 application on US data 1970Q1-2019Q4: every figure and table (shrinkage, kernels, GIRFs, asymmetries, sub-samples, forecasts, evaluation)."),
            ("guide.html", "User guide", "How to write code with gpvar: data, model specification, estimation, GIRFs, forecasts, figures, tables, with runnable snippets and the full signature reference."),
        ]) + "</div>"
    overview += md(section(1)) + md(section(2)) + md(section(3))
    overview += "<h2>Headline results</h2>" + md(
        "The library reproduces the paper's Section 6 analysis on real data. Average responses to a positive one-standard-deviation "
        "uncertainty shock (posterior median, 68 % bands) compared with a linear BVAR, and the sign asymmetry that is the paper's central finding:")
    overview += figure("gpvar8_fig07_girf_vs_bvar_focus", "Average GIRFs of the focus variables, GP-VAR-8 (orange) vs. BVAR-8 (grey).")
    overview += figure("gpvar8_fig09_sign_asymmetry_focus", "Responses to positive (orange) and negative (blue) shocks; the mirrored negative response (grey) would coincide with the orange line in a linear model.")
    overview += md(section(9)) + md(section(10)) + md(section(11)) + md(section(12)) + md(section(13)) + md(section(14))
    open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8").write(
        page("Overview", overview, "index.html", "A faithful, self-contained Python implementation of the GP-VAR with stochastic volatility of Hauzenberger, Huber, Marcellino and Petz (2022)"))

    # ---------------- concepts
    concepts = open(os.path.join(DOCS, "CONCEPTS.md"), encoding="utf-8").read()
    open(os.path.join(DOCS, "concepts.html"), "w", encoding="utf-8").write(
        page("Concepts", md(concepts), "concepts.html", "The Gaussian process VAR explained: from kernels to generalized impulse responses"))

    # ---------------- results
    res = ["<p>Every figure and table produced by <code>examples/03_us_uncertainty_gpvar8.py</code>, <code>02_simulation_study.py</code>, "
           "<code>01_gp_regression_illustration.py</code> and <code>04_forecast_evaluation.py</code> on the bundled data. Responses are in "
           "standard-deviation units of the standardised data unless stated otherwise; bands are 68 % posterior credible sets; "
           "figures are also available as vector PDFs and tables as LaTeX (booktabs) and CSV in <code>outputs/</code> of the repository.</p>"]
    res.append(md(section(5).replace("## 5. Real economic example", "## Real economic example")))
    res.append("<h2>Non-focus variables (Online Appendix C)</h2>")
    res.append(figure("gpvar8_figC4_girf_vs_bvar_nonfocus", "Figure C.4 analogue: GIRFs of AWH, CPI, AHE and FFR, GP-VAR-8 vs. BVAR-8."))
    res.append(figure("gpvar8_figC9_sign_asymmetry_nonfocus", "Figure C.9 analogue: sign asymmetries for the non-focus variables."))
    res.append(figure("gpvar8_figC10_size_asymmetry_nonfocus", "Figure C.10 analogue: size asymmetries for the non-focus variables."))
    res.append(figure("gpvar8_figC5_subsamples_nonfocus", "Figure C.5 analogue: sub-sample GIRFs of the non-focus variables."))
    res.append(figure("gpvar8_figC7_time_variation_nonfocus", "Figure C.7 analogue: yearly GIRFs of the non-focus variables."))
    res.append("<h2>Further tables</h2>")
    for name, cap in [("gpvar8_data_description", "Data description (Table B.1 analogue)."),
                      ("gpvar8_descriptives", "Descriptive statistics of the GP-VAR-8 data set."),
                      ("gpvar8_Q", "Free elements of the contemporaneous matrix Q (horseshoe prior)."),
                      ("gpvar8_girf_negative_horizons", "Average GIRFs to a negative one-standard-deviation shock."),
                      ("gpvar8_girf_peaks_by_period", "Peak responses to a positive shock by sub-sample."),
                      ("gpvar8_state_dependence", "State dependence: dispersion of history-specific responses."),
                      ("gpvar8_forecast_quantiles", "Predictive quantiles from the end of the sample (original units)."),
                      ("table02b_crps", "Average CRPS in the recursive evaluation (lower is better).")]:
        t = table_html(name)
        if t:
            res.append(f"<h3>{cap}</h3>" + t)
    res.append("<h2>MCMC diagnostics</h2>")
    res.append(figure("gpvar8_mcmc_traces", "Traces of the kernel hyperparameters (own lags blue, other lags green) and of the SV persistence per equation."))
    res.append(md(section(6).replace("## 6. Simulation study", "## Simulation study")))
    res.append(figure("fig03b_simulation_volatility", "Posterior of the structural error standard deviations versus the truth in the simulation."))
    res.append(md(section(7).replace("## 7. GP regression illustration", "## GP regression illustration")))
    res.append(figure("fig01_gp_inflation_trend", "Figure 1 analogue: inflation on a linear time trend for four kernels."))
    res.append(figure("figA1_gp_persistence_kernel", "Figure A.1 analogue: the linear persistence kernel."))
    res.append(md(section(8).replace("## 8. Forecast evaluation", "## Forecast evaluation")))
    res.append(figure("forecast_cumulative_lpbf_h1", "Cumulative joint LPBF vs. the BVAR-8 at h = 1."))
    open(os.path.join(DOCS, "results.html"), "w", encoding="utf-8").write(
        page("Results", "\n".join(res), "results.html", "Uncertainty shocks in the US economy, 1970Q1-2019Q4: all outputs of the real-data application"))

    # ---------------- guide and methodology
    guide = open(os.path.join(DOCS, "USER_GUIDE.md"), encoding="utf-8").read()
    guide = re.sub(r"^# .*\n", "", guide, count=1)
    open(os.path.join(DOCS, "guide.html"), "w", encoding="utf-8").write(
        page("User guide", md(guide), "guide.html", "Syntax reference: how to write code with gpvar"))
    meth = open(os.path.join(DOCS, "METHODOLOGY.md"), encoding="utf-8").read()
    meth = re.sub(r"^# .*\n", "", meth, count=1)
    open(os.path.join(DOCS, "methodology.html"), "w", encoding="utf-8").write(
        page("Methodology", md(meth), "methodology.html", "What gpvar implements and how each step maps to the paper"))
    print("site built in", DOCS)


if __name__ == "__main__":
    build()
