#  <p align="center">:robot:  Data09:robot:  </p> 

> <p align="center">Implementation of a web service that classifies images using convolutional neural networks</p> 

## <p align="center">:smiley_cat: Мова читання - українська. Основна інформація про проект</p>

Застосунок "Фудамент" - це веб-сервіс для класифікації зображень, заснований на використанні згорткових нейронних мереж (CNN) та оптимізований для роботи із датаcетоv [cifar-10](https://www.kaggle.com/c/cifar-10). . Проект дозволяє користувачам завантажувати зображення, які класифікуються за допомогою попередньо тренованих моделей, таких VGG16. Сервіс підтримує широкий спектр функцій, включаючи завантаження зображень, тренування моделей, перевірку якості зображень, а також детальне відображення результатів класифікації. Особлива увага приділяється оптимізації моделей для зменшення їх розміру та підвищення точності класифікації, що робить сервіс ефективним інструментом для обробки зображень з cifar-10.

## <p align="center">:e-mail: Команда проекту</p>

[Volodymyr Rizun](https://github.com/VolodymyrRiz) [Linkedin](https://www.linkedin.com/in/volodymyr-rizun-40103219b/) 

[Vladyslav Skopenko](https://github.com/VladSkopenko) [Linkedin](https://www.linkedin.com/in/vladyslav-skopenko/) 

[Viktoriia Kalachova](https://github.com/ViKalachova) [Linkedin](https://www.linkedin.com/in/viktoriia-kalachova-746228282/) 

[Borys Kuchyn](https://github.com/Zpfamily) [Linkedin](https://www.linkedin.com/in/borys-kuchyn-1316581a4/) 

[Dmytro Ostrenko](https://github.com/Dmytro-Ostrenko) [Linkedin](https://www.linkedin.com/in/volodymyr-rizun-40103219b/) 

## <p align="center">:open_file_folder: Технології, які були використані у проекті</p> 

## <p align="center">:scroll: Головні функції застосунку</p>

1. **Завантаження зображень для класифікації**  
Користувач може завантажувати зображення для класифікації, які потім обробляються нейронною мережею для визначення категорії.

2. **Взаємодія з чотирма натренованими нейронними мережами**  
Користувач має можливість обрати одну з чотирьох архітектур нейронних мереж і налаштувати поріг впевненості моделі у межах від 0 до 1. В залежності від цього рівня, застосовуються різні фільтри для покращення деталізації зображення.

3. **Завантаження та аналіз натренованих нейронних мереж**  
Користувач може завантажити натреновану нейронну мережу, ознайомитися з інформацією про кожну мережу, а також переглянути графіки тренування на тестових та валідаційних даних.

4. **Контейнеризація з Docker**  
Проект повністю контейнеризований за допомогою [Docker](https://www.docker.com/), що забезпечує легкість розгортання та інтеграції в різних середовищах.

Ці функції забезпечують зручний і гнучкий інтерфейс для роботи з нейронними мережами та зображеннями, а також спрощують розгортання додатку.

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


### Огляд

Цей проект реалізує класифікацію зображень з датасету CIFAR-10, використовуючи чотири різні архітектури нейронних мереж. Нижче наведені типи моделей, що використовуються, та їх точність:

1. **VGG16** (точність 90%)
2. **AlexNet** (точність 89%)
3. **LeNet** (точність 79%)
4. **MobileNet** (точність 86%)

### Опис моделей

### 1. VGG16
[VGG16](https://keras.io/api/applications/vgg/) – це одна з найпопулярніших архітектур згорткових нейронних мереж (CNN), яка складається з 16 шарів. Основна ідея VGG16 полягає у використанні невеликих фільтрів розміром 3x3 у всіх згорткових шарах, що дозволяє зберігати просторову інформацію та досягати високої точності. Модель закінчується кількома повнозв’язними шарами та шаром Softmax для класифікації.

### 2. AlexNet
[AlexNet](https://www.kaggle.com/code/blurredmachine/alexnet-architecture-a-complete-guide)  – це архітектура CNN, яка вперше продемонструвала силу глибокого навчання на конкурсі ImageNet у 2012 році. Модель складається з 8 навчальних шарів, включаючи 5 згорткових і 3 повнозв’язних. Особливості AlexNet включають використання великих фільтрів на початкових шарах та Dropout для запобігання перенавчанню.

### 3. LeNet
[LeNet](https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/) – одна з перших CNN архітектур, розроблена [Яном ЛеКуном](https://yann.lecun.com/) для розпізнавання рукописних цифр. Вона складається з чергування згорткових і підвибіркових (Pooling) шарів, завершуючись кількома повнозв’язними шарами. LeNet є простою в реалізації, але менш потужною порівняно з іншими сучасними архітектурами.

### 4. MobileNet
[MobileNet](https://keras.io/api/applications/mobilenet/)  – це легка архітектура CNN, спеціально розроблена для мобільних та вбудованих систем. Вона використовує глибокі згортки (Depthwise Separable Convolutions), що значно зменшує кількість параметрів і обчислень, роблячи її ефективною для застосувань з обмеженими ресурсами.

## Використання

Проект "Фундамент" дозволяє користувачу вибрати одну з чотирьох моделей для класифікації зображень з датасету cifar-10. Попередньо натреновані моделі збережені  у форматі `.keras`, що дозволяє їх повторне використання без необхідності перенавчання.



