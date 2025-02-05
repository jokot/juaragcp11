import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part

def generate_bouquet_image(
        project_id: str, 
        location: str, 
        prompt: str
    ) -> vertexai.preview.vision_models.ImageGenerationResponse:

    vertexai.init(
        project = "",
        location = ""
    )

    model = ImageGenerationModel.from_pretrained("imagegeneration@002")

    images = model.generate_images(
        prompt = prompt
    )

    images[0].save(location = "bouquet.jpeg")

    return images



def analyze_bouquet_image(project_id: str, location: str, image_path: str) -> str:
    vertexai.init(
        project_id = project_id,
        location = location
    )

    multimodal_model = GenerativeModel("gemini-pro-vision")

    response = multimodal_model.generate_content(
        [
            Part.from_uri(
                "gs://generativeai-downloads/images/scones.jpg", mime_type="image/jpeg"
            ),
            
            "Generate birthday wishes from this image"
        ],
        stream = True
    )

    return response.text

project_id = "project-id"
location = "REGION"
file_name = "bouquet.jpeg"

generate_bouquet_image(
    project_id=project_id,
    location=location,
    prompt= "Create an image containing a bouquet of 2 sunflowers and 3 roses"
)

analyze_bouquet_image(
    project_id=project_id,
    location=location,
    image_path=file_name
)

