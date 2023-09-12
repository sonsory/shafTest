from odmantic import Model


class BookModel(Model):
    keyword: str
    publisher: str
    # discount: int
    price: int
    image: str

    class Config:
        collection = "books"  # 해당 이름의 콜렉션이 없을 경우, 새로 만든다.
