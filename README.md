#  <p align="center">:robot:  Data09:robot:  </p> 

> <p align="center">Implementation of a web service that classifies images using convolutional neural networks</p> 

## <p align="center">:smiley_cat: Мова читання - українська. Основна інформація про проект</p>

Застосунок "Фудамент" - це веб-сервіс для класифікації зображень, заснований на використанні згорткових нейронних мереж (CNN) та оптимізований для роботи із датаcетоv [cifar-10](https://www.kaggle.com/c/cifar-10). . Проект дозволяє користувачам завантажувати зображення, які класифікуються за допомогою попередньо тренованих моделей, таких VGG16. Сервіс підтримує широкий спектр функцій, включаючи завантаження зображень, тренування моделей, перевірку якості зображень, а також детальне відображення результатів класифікації. Особлива увага приділяється оптимізації моделей для зменшення їх розміру та підвищення точності класифікації, що робить сервіс ефективним інструментом для обробки зображень з cifar-10.

## <p align="center">:open_file_folder: Технології, які були використані</p> 

## <p align="center">:scroll: Головні функції застосунку</p>

## <p align="center">:bookmark_tabs: Інструкція з встановлення та користування ПЗ «Фундамент»</p>

### Передумови
Переконайтеся, що на вашому комп'ютері встановлено Python версії 3.11 або новіше. Ви можете завантажити Python з [офіційного сайту](https://www.python.org/downloads/).

### Встановлення

Перед тим, як ви почнете використовувати веб-інтерфейс "Фундамент" із штучними нейронними мережами, для розпізнавання картинок, Вам потрібно встановити його. Дотримуйтесь цих кроків:

1. **Клонування репозиторію на свій комп'ютер** :white_check_mark::   

```
git clone https://github.com/Dmytro-Ostrenko/Data09
```

2. **Перехід у каталог проєкту** :white_check_mark::  

```
cd Data09
```

3. **Встановлення залежностей з `requirements.txt`** :white_check_mark::
   
```
pip install -r requirements.txt 
```

4. **Створення та активація віртуального середовища** :white_check_mark::
   
```
python -m venv venv
source venv/bin/activate  # Для Linux/MacOS
.\venv\Scripts\activate   # Для Windows
```

5. **Оновлення залежностей (опціонально)** :white_check_mark::

```
pip install --upgrade -r requirements.txt
```

6. **Запуск застосунку** :white_check_mark::

```
cd Data09/data9
python manage.py runserver
```


Також можна швидко [скачати](https://github.com/Dmytro-Ostrenko/Data09/archive/refs/heads/main.zip)  чи [клонувати](https://github.com/Dmytro-Ostrenko/Data09.git).

## <p align="center">:card_index: Типи нейронних мереж, що використані у проекті </p>


