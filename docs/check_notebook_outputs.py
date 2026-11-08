"""Check that the published notebook tutorials include their plots and videos."""

import sys
from html.parser import HTMLParser
from pathlib import Path

PLOTS = {
    "advanced/model_parameters": 4,
    "basics/00_overview": 2,
    "basics/01a_extracellular_stim_and_rec": 3,
    "basics/01b_precomputed_potentials": 3,
    "basics/01d_thresholding": 4,
    "basics/02_branched_morphologies": 8,
    "basics/03_networks": 1,
    "basics/04_gradient_descent": 2,
}
NOTEBOOK_WARNINGS = (
    "TqdmWarning: IProgress not found",
    "FigureCanvasAgg is non-interactive",
    "Warning: no DISPLAY environment variable",
)


class NotebookMedia(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.depth = 0
        self.output_depth = None
        self.images = []
        self.videos = 0
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "div":
            self.depth += 1
            if "cell_output" in attributes.get("class", "").split():
                self.output_depth = self.depth
        elif self.output_depth is not None:
            if tag == "img" and (source := attributes.get("src", "")).startswith(
                "../_images/"
            ):
                self.images.append(source)
            elif tag == "video":
                self.videos += 1

    def handle_endtag(self, tag):
        if tag == "div":
            if self.depth == self.output_depth:
                self.output_depth = None
            self.depth -= 1


def check_notebook_outputs(html_dir: Path):
    errors = []
    for name, minimum in PLOTS.items():
        page = html_dir / f"{name}.html"
        images = NotebookMedia(page.read_text(encoding="utf-8")).images
        if len(images) < minimum:
            errors.append(f"{page.name}: expected {minimum} plots, found {len(images)}")
        for image in images:
            if not (page.parent / image).is_file():
                errors.append(f"{page.name}: missing plot asset {image}")

    multicontact = html_dir / "basics" / "01c_multicontact.html"
    if NotebookMedia(multicontact.read_text(encoding="utf-8")).videos < 3:
        errors.append(f"{multicontact.name}: expected three rendered videos")

    for page in html_dir.rglob("*.html"):
        html = page.read_text(encoding="utf-8")
        errors.extend(
            f"{page.relative_to(html_dir)}: {warning}"
            for warning in NOTEBOOK_WARNINGS
            if warning in html
        )
    if errors:
        raise SystemExit("\n".join(errors))
    print("Verified plots, videos, and headless notebook setup in the rendered docs.")


if __name__ == "__main__":
    check_notebook_outputs(Path(sys.argv[1]))
