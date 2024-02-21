import pytest
from aiogram.filters import Command
from aiogram.methods import SendMessage
from aiogram.types import ReplyKeyboardMarkup

from bot import command_handler
from bot import continue_yes
from bot import continue_no
from bot import continue_unexpected
from bot import ml_method
from bot import dl_method
from bot import unexpected_method
from bot import quit_method
from bot import generate_keyboard
from bot import States

from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE


# Тест функции generate_keyboard
def test_generate_keyboard():
    # Создаем список кнопок
    button_names = ["Да", "Нет", "Отмена"]

    # количество кнопок в ряду
    buttons_in_row = 3

    # Вызываем функцию generate_keyboard
    keyboard = generate_keyboard(button_names, buttons_in_row)

    # Проверяем тип возвращаемого объекта
    assert isinstance(keyboard, ReplyKeyboardMarkup)

    # Проверяем, что количество кнопок в ряду не больше buttons_in_row
    for row in keyboard.keyboard:
        assert len(row) <= buttons_in_row

    # Проверяем текст кнопок
    expected_button_texts = ["Да", "Нет", "Отмена"]
    for i, row in enumerate(keyboard.keyboard):
        for j, button in enumerate(row):
            assert button.text == expected_button_texts[i * 2 + j]


def test_Responses():
    # Создаем экземпляр класса States
    responses = States()
    expected_attr = ['yes_no', 'type_detection', 'ml_foto']

    # Проверяем, что состояния имеют ожидаемые атрибуты
    for attribute in expected_attr:
        assert hasattr(responses, attribute)

    # Проверяем, что каждое состояние уникально
    for i, attribute in enumerate(expected_attr):
        if len(expected_attr[i + 1:]) > 1:
            for matched_attr in expected_attr[i + 1:]:
                assert getattr(responses, attribute) != matched_attr


@pytest.mark.asyncio
async def test_command_handler():
    requester = MockedBot(
        request_handler=MessageHandler(command_handler,
                                       Command(commands=["start"])))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Это бот по детекции Дипфейков на фотографии! \
             \nХотите попробовать?"


@pytest.mark.asyncio
async def test_continue_yes():
    requester = MockedBot(
        request_handler=MessageHandler(continue_yes,
                                       state=States.yes_no))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="да"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Отличный выбор! \nВыберите метод или нажмите Отмена для выхода"


@pytest.mark.asyncio
async def test_continue_no():
    requester = MockedBot(
        request_handler=MessageHandler(continue_no,
                                       state=States.yes_no))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="нет"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Жаль! \nДо встречи в следующий раз"


@pytest.mark.asyncio
async def test_continue_unexpected():
    requester = MockedBot(
        request_handler=MessageHandler(continue_unexpected,
                                       state=States.yes_no))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object())
    answer_message = calls.send_message.fetchone().text
    assert answer_message == ("Введите 'Да' для продолжения или 'Нет' для выхода")


@pytest.mark.asyncio
async def test_ml_method():
    requester = MockedBot(
        request_handler=MessageHandler(ml_method,
                                       state=States.type_detection)
    )
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="ml"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Machine Learning - отличный выбор!\nДобавьте фотографию"


@pytest.mark.asyncio
async def test_dl_method():
    requester = MockedBot(
        request_handler=MessageHandler(dl_method,
                                       state=States.type_detection)
    )
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="dl"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Deep Learning. \
            \nК сожалению, данный метод пока еще в разработке.\
            \nВведите 'ML' для продолжения или 'Отмена' для выхода"


@pytest.mark.asyncio
async def test_unexpected_method():
    requester = MockedBot(
        request_handler=MessageHandler(unexpected_method,
                                       state=States.type_detection)
    )
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object())
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Выберите метод или нажмите Отмена для выхода"


@pytest.mark.asyncio
async def test_quit_method():
    requester = MockedBot(
        request_handler=MessageHandler(quit_method,
                                       state=States.type_detection)
    )
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="Отмена"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Жаль! \nДо встречи в следующий раз"
