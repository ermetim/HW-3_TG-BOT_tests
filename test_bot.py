import pytest
from aiogram.filters import Command
from aiogram.methods import SendMessage

from bot import command_handler
from bot import continue_yes
from bot import continue_no
from bot import continue_unexpected
from bot import ml_method
from bot import dl_method
from bot import unexpected_method
from bot import quit_method
from bot import States

from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE


@pytest.mark.asyncio
async def test_command_handler():
    requester = MockedBot(
        request_handler=MessageHandler(command_handler,
                                       Command(commands=["start"])))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Это бот по детекции Дипфейков на фотографии! \
                         \nХотите попробовать? \nВведите Да или Нет"


@pytest.mark.asyncio
async def test_continue_yes():
    requester = MockedBot(
        request_handler=MessageHandler(continue_yes,
                                       state=States.state))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="да"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Отличный выбор! \nКакой метод ML или DL.\
                        \nНажмите Отмена для выхода"


@pytest.mark.asyncio
async def test_continue_no():
    requester = MockedBot(
        request_handler=MessageHandler(continue_no,
                                       state=States.state))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="отмена"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Жаль! \nДо встречи в следующий раз"


@pytest.mark.asyncio
async def test_continue_unexpected():
    requester = MockedBot(
        request_handler=MessageHandler(continue_unexpected,
                                       state=States.state))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object())
    answer_message = calls.send_message.fetchone().text
    assert answer_message == ("Введите 'Да' для продолжения или 'Нет' для выхода")


@pytest.mark.asyncio
async def test_ml_method():
    requester = MockedBot(request_handler=MessageHandler(ml_method,
                                                         state=States.state_1))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="ml"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Machine Learning - отличный выбор!\
                        \nДобавьте фотографию"


@pytest.mark.asyncio
async def test_dl_method():
    requester = MockedBot(request_handler=MessageHandler(dl_method,
                                                         state=States.state_1))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="dl"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Deep Learning. \
            \nК сожалению, данный метод пока еще в разработке.\
            \nВведите 'ML' для продолжения или 'Отмена' для выхода"


@pytest.mark.asyncio
async def test_unexpected_method():
    requester = MockedBot(request_handler=MessageHandler(unexpected_method,
                                                         state=States.state_1))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object())
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Введите ML или DL для продолжения \
                        или 'Отмена' для выхода"


@pytest.mark.asyncio
async def test_quit_method():
    requester = MockedBot(request_handler=MessageHandler(quit_method,
                                                         state=States.state_1))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="Отмена"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Жаль! \nДо встречи в следующий раз"
