def validate_pill_color(color: str) -> bool:
    """
    Check validity of a color name string.
    Valid color names' list is ["하양", "노랑", "주황", "분홍", "빨강", "갈색", "연두", "초록", "청록", "파랑", "남색", "자주", "보라", "회색", "검정", "투명"].

    :param color: Color name of the pill from user input to search
    :return: Validity of the color name. If the color name is included in
    valid color names' list, returns true. Otherwise, returns false.
    """
    is_valiid = color in ["하양", "노랑", "주황", "분홍", "빨강", "갈색", "연두", "초록", "청록", "파랑", "남색", "자주", "보라", "회색", "검정", "투명"]
    print(f'@ validate color: {color}, {is_valiid}')
    return is_valiid


def validate_pill_shape(shape: str) -> bool:
    """
    Check validity of a shape name string.
    Valid shape names' list is ["원형", "타원형", "장방형", "반원형", "삼각형", "사각형", "마름모형", "오각형", "육각형", "팔각형", "기타"].
    If the shape is not included in the list, shape will be replaced with "기타" at the searching step.

    :param shape: Shape name of the pill to search
    :return: Validity of the shape name. If the shape name is included in
    valid shape names' list, returns true. Otherwise, returns false.
    """
    is_valid = shape in ["원형", "타원형", "장방형", "반원형", "삼각형", "사각형", "마름모형", "오각형", "육각형", "팔각형", "기타"]
    print(f'@ validate shape: {shape}, {is_valid}')
    return is_valid


def validate_age(age: str) -> bool:
    """
    Validate age string. Valid age is non-empty string.
    :param age: User's age.
    :return: Validity of the age. if age is non-empty string, returns true. Otherwise, returns false.
    """
    is_valid = bool(age)
    print(f'@ validate age: {age}, {is_valid}')
    return is_valid


def validate_gender(gender: str) -> bool:
    """
    Validate gender string. Valid gender is non-empty string.
    :param gender: User's gender.
    :return: Validity of the gender. if gender is non-empty string, returns true. Otherwise, returns false.
    """
    is_valid = bool(gender)
    print(f'@ validate gender: {gender}, {is_valid}')
    return is_valid


tools = (
    validate_pill_color,
    validate_pill_shape,
    validate_age,
    validate_gender
)
