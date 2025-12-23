"""PDF to image conversion module using PyMuPDF."""

import fitz  # PyMuPDF
import numpy as np
import cv2
from typing import List


class PDFConversionError(Exception):
    """Raised when PDF conversion fails."""
    pass


def pdf_to_images(pdf_path: str) -> List[np.ndarray]:
    """
    Convert multi-page PDF to list of images preserving original dimensions.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of numpy arrays (BGR format for OpenCV), one per page

    Raises:
        FileNotFoundError: If PDF doesn't exist
        PDFConversionError: If PDF is corrupted or cannot be read
    """
    try:
        doc = fitz.open(pdf_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise PDFConversionError(f"Failed to open PDF: {str(e)}")

    images = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Get page at original resolution (matrix=fitz.Identity preserves dimensions)
            # For higher quality, use matrix=fitz.Matrix(2, 2) for 2x scaling
            pix = page.get_pixmap(matrix=fitz.Identity)

            # Convert to numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            # Convert RGB to BGR for OpenCV
            if pix.n == 3:  # RGB
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pix.n == 4:  # RGBA
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:  # Grayscale
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

            images.append(img_bgr)

    except Exception as e:
        raise PDFConversionError(f"Failed to convert PDF page: {str(e)}")
    finally:
        doc.close()

    if not images:
        raise PDFConversionError(f"PDF has no pages: {pdf_path}")

    return images


def is_pdf(file_path: str) -> bool:
    """
    Check if file is a PDF based on extension.

    Args:
        file_path: Path to file

    Returns:
        True if file has .pdf extension (case-insensitive)
    """
    return file_path.lower().endswith('.pdf')


def load_image_or_pdf(file_path: str) -> List[np.ndarray]:
    """
    Load images from PDF or image file.

    Args:
        file_path: Path to PDF or image file

    Returns:
        List of BGR images (single item for image files, multiple for PDFs)

    Raises:
        FileNotFoundError: If file doesn't exist
        PDFConversionError: If PDF conversion fails
        ValueError: If image file cannot be read
    """
    if is_pdf(file_path):
        return pdf_to_images(file_path)
    else:
        # Load as image
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Failed to load image: {file_path}")
        return [img]
