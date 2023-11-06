"""
FastAPI app for face verification

API endpoints:
    - /verify
      POST:
        Request body: JSON
        payload: {
            "image": "base64 encoded image",
            "known_images": ["base64 encoded image1", "base64 encoded image2", ...],
          }
"""
import asyncio
import base64
import imghdr
import io
import logging
import logging.config
import pkg_resources
import os
import tempfile


import face_recognition
import numpy as np
from PIL import Image
from pydantic import BaseModel, constr, conlist

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


__version__ = pkg_resources.get_distribution("face_verification_api").version
__title__ =  pkg_resources.get_distribution("face_verification_api").project_name


_logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(
    title=__title__,
    description="Face verification API",
    version=__version__,
    docs_url="/",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# class FaceVerificationRequest(BaseModel):
    # """Face verification request schema"""

    # image: constr(min_length=1)
    # kwown_images: conlist(constr(min_length=1), min_length=1)


class FaceVerificationResponse(BaseModel):
    """Face verification response schema"""

    is_verified: bool
    reason: str = None

@app.post("/verify", response_model=FaceVerificationResponse, status_code=status.HTTP_200_OK)
async def verify(
    image_file: UploadFile = File(...),
    known_image_files: list[UploadFile] = File(...),
) -> FaceVerificationResponse:
    """Verify if the given image is of the given person

    Args:
        image_file (UploadFile): image file
        known_image_files (list[UploadFile]): list of known images

    Returns:
        FaceVerificationResponse: response
    """
    try:
        is_verified, raison = await _verify(image_file, known_image_files)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return FaceVerificationResponse(is_verified=is_verified, reason=raison)



async def _verify(
    image_file: UploadFile,
    known_image_files: list[UploadFile],
) -> tuple[bool, str]:
    """Verify if the given image is of the given person

    Args:
        image_file : image file
        known_image_files (list): list of known images

    Returns:
        bool: True if the given image is of the given person, False otherwise
    """

    # Load the images
    image_file = await image_file.read()
    known_image_files = await asyncio.gather(*[known_image_file.read() for known_image_file in known_image_files])

    # Convert the images to numpy arrays
    image = face_recognition.load_image_file(io.BytesIO(image_file))
    known_images = [face_recognition.load_image_file(io.BytesIO(known_image_file)) for known_image_file in known_image_files]


    # Get the encodings
    image_encodings = face_recognition.face_encodings(image)
    if len(image_encodings) == 0:
        return False, "No face found in the given image"
    image_encoding = image_encodings[0]

    known_encodings = []
    for known_image in known_images:
        known_encodings.extend(face_recognition.face_encodings(known_image))
    if len(known_encodings) == 0:
        return False, "No face found in the known images"



    # Compare the encodings
    results = face_recognition.compare_faces(known_encodings, image_encoding)

    # Return True if the given image is of the given person, False otherwise
    return any(results), "No match found" if not any(results) else None


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="http://localhost", port=8000)
