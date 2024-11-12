def validate_pill_color(color: str) -> bool:
    """
    Valid color names' list is ["하양", "노랑", "주황", "분홍", "빨강", "갈색", "연두", "초록", "청록", "파랑", "남색", "자주", "보라", "회색", "검정", "투명"].

    :param color: Color name of the pill from user input to search
    :return: Validity of the color name. If the color name is included in
    valid color names' list, returns true. Otherwise, returns false.
    """
    return color in ["하양", "노랑", "주황", "분홍", "빨강", "갈색", "연두", "초록", "청록", "파랑", "남색", "자주", "보라", "회색", "검정", "투명"]


def validate_pill_shape(shape: str) -> bool:
    """
    Valid shape names' list is ["원형", "타원형", "장방형", "반원형", "삼각형", "사각형", "마름모형", "오각형", "육각형", "팔각형", "기타"].

    :param shape: Shape name of the pill to search
    :return: Validity of the shape name. If the shape name is included in
    valid shape names' list, returns true. Otherwise, returns false.
    """
    return shape in ["원형", "타원형", "장방형", "반원형", "삼각형", "사각형", "마름모형", "오각형", "육각형", "팔각형", "기타"]


def validate_query_input(query: str, age: str, gender: str) -> bool:
    """
    Validate chatbot's form filling for web search.
    
    :param query: 사용자가 웹에서 찾고싶어 하는 검색어 
    :param age: 사용자의 연령대
    :param gender: 사용자의 성별
    :return: 양식이 완성되었으면 true, 아니면 false`
    """
    return bool(query and age and gender)


tools = (
    validate_pill_color,
    validate_pill_shape,
    validate_query_input
)
