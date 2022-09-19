from pydantic import BaseModel, validate_arguments


class User(BaseModel):
    id: int
    name: str


@validate_arguments
def func(data: User):
    print(data)


if __name__ == '__main__':
    data = {"id": 1, "name": "wangzhenxiong"}
    func(data)
