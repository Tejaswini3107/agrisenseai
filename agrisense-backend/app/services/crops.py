from enum import Enum


class CropType(str, Enum):
    RICE = "rice"
    WHEAT = "wheat"
    COTTON = "cotton"
    MAIZE = "maize"
    TURMERIC = "turmeric"
    SUGARCANE = "sugarcane"
    SOYBEAN = "soybean"
    POTATO = "potato"
    ONION = "onion"
    GARLIC = "garlic"


CROP_ALIASES = {
    "paddy": CropType.RICE.value,
    "rice": CropType.RICE.value,
    "wheat": CropType.WHEAT.value,
    "cotton": CropType.COTTON.value,
    "maize": CropType.MAIZE.value,
    "corn": CropType.MAIZE.value,
    "turmeric": CropType.TURMERIC.value,
    "sugarcane": CropType.SUGARCANE.value,
    "soybean": CropType.SOYBEAN.value,
    "potato": CropType.POTATO.value,
    "onion": CropType.ONION.value,
    "garlic": CropType.GARLIC.value,
    "tomato": CropType.SUGARCANE.value,
}

CROP_CATALOG = [
    {"key": crop.value, "label": crop.value.title()} for crop in CropType
]


def normalize_crop(crop_name: str | None) -> str:
    if not crop_name:
        return CropType.RICE.value

    return CROP_ALIASES.get(crop_name.strip().lower(), crop_name.strip().lower())


def is_supported_crop(crop_name: str | None) -> bool:
    return normalize_crop(crop_name) in CROP_ALIASES.values()


def get_crop_catalog():
    return CROP_CATALOG
