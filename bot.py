import asyncio
import logging
import os
import pickle
import cv2
import numpy as np

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message


from functions import ImageProcessing

MODEL_NAMES = ['lgbm_model_hog_best.pkl',
               'lgbm_model_pic_best.pkl',
               'rfc_model_hog_best.pkl',
               'rfc_model_pic_best.pkl',
               ]

# fake token
BOT_TOKEN = '123456789:AABBCCDDEEFFaabbccddeeff-1234567890'

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


class TestCallbackData(CallbackData, prefix="test_callback_data"):
    id: int
    name: str


dp = Dispatcher(storage=MemoryStorage())


class States(StatesGroup):
    state = State()
    state_1 = State()
    state_2 = State()


@dp.message(Command(commands=["start"]))
async def command_handler(message: types.Message, state: FSMContext) -> None:
    await message.answer("Это бот по детекции Дипфейков на фотографии! \
                         \nХотите попробовать? \nВведите Да или Нет")
    await state.set_state(States.state)


@dp.message(F.text.lower() == "да", States.state)
async def continue_yes(message: types.Message,
                       state: FSMContext) -> None:
    await message.reply("Отличный выбор! \nКакой метод ML или DL.\
                        \nНажмите Отмена для выхода")
    await state.set_state(States.state_1)


@dp.message(F.text.lower() == "нет", States.state)
async def continue_no(message: types.Message,
                      state: FSMContext) -> None:
    await message.reply("Жаль! \nДо встречи в следующий раз")
    await state.clear()


@dp.message(States.state)
async def continue_unexpected(message: types.Message) -> None:
    await message.reply("Введите 'Да' для продолжения или 'Нет' для выхода")


@dp.message(F.text.lower() == "ml", States.state_1)
async def ml_method(message: types.Message,
                    state: FSMContext) -> None:
    await message.reply("Machine Learning - отличный выбор!\
                        \nДобавьте фотографию")
    await state.set_state(States.state_2)


@dp.message(F.text.lower() == "dl", States.state_1)
async def dl_method(message: types.Message) -> None:
    await message.reply("Deep Learning. \
            \nК сожалению, данный метод пока еще в разработке.\
            \nВведите 'ML' для продолжения или 'Отмена' для выхода")


@dp.message(States.state_1)
async def unexpected_method(message: types.Message) -> None:
    await message.reply("Введите ML или DL для продолжения \
                        или 'Отмена' для выхода")


@dp.message(F.text.lower() == "отмена", States.state_1)
async def quit_method(message: types.Message, state: FSMContext) -> None:
    await message.reply("Жаль! \nДо встречи в следующий раз")
    await state.clear()


@dp.message(F.photo, States.state_2)
async def ml_photo(message: Message, bot: Bot, state: FSMContext):

    await message.answer("Ваша фотография получена. \
                         \nИдет процесс обработки....")

    # сохраняем в формате class '_io.BytesIO'
    bytes_io_object = await bot.download(message.photo[-1])

    # переводим в numpy
    np_photo = np.frombuffer(bytes_io_object.getvalue(), dtype=np.uint8)

    # кодируем в фотографию
    img = cv2.imdecode(np_photo, cv2.IMREAD_COLOR)

    # сохраняем в корень
    # cv2.imwrite('pic.jpg', img)

    image, face_image, hog_image = ImageProcessing().transform_image(img)
    models_path = os.path.join('models', 'ML')
    predictions = np.array([])
    for model_name in MODEL_NAMES:
        model = pickle.load(open(os.path.join(models_path, model_name), 'rb'))
        if 'hog' in model_name:
            predictions = np.append(predictions,
                                    model.predict_proba([hog_image.ravel()])[:, 1])
        else:
            predictions = np.append(predictions,
                                    model.predict_proba([image.ravel()])[:, 1])
    avg_pred = np.mean(predictions)
    await message.answer(f"Ваша фотография является Фейком \
                         с вероятностью {round(avg_pred,3) * 100}%")

    await state.clear()


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
