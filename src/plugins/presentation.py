from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from pptx import Presentation


class PresentationPlugin:
    """The Presentation Plugin can be used to create presentations decks and slides."""

    @kernel_function(description="Create a presentation deck in a PDF format.")
    def create_presentation(self, 
                            title: Annotated[str, "The title of the presentation"],
                            content: Annotated[str, "The content of the decks"]) -> Annotated[str, "Create a presentation."]:

        # Create a presentation object
        prs = Presentation(pptx='green.pptx')

        # Split the content into slides
        slides_content = content.split("\n\n")

        for slide_content in slides_content:
            # Add a slide with the specified layout
            slide_layout = prs.slide_layouts[0]
            slide = prs.slides.add_slide(slide_layout)

            # Add title and content to the slide
            title, *body = slide_content.split("\n")
            slide.shapes.title.text = title
            slide.placeholders[1].text = "\n".join(body)

        # Save the presentation
        output_path = "presentation.pptx"
        prs.save(output_path)

        return output_path
