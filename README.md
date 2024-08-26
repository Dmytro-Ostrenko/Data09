#  <p align="center">:robot:  Data09:robot:  </p> 

> <p align="center">Implementation of a web service that classifies images using convolutional neural networks</p> 

---
## <p align="center">:smiley_cat: Мова читання - українська. Основна інформація про проект</p>
---

Застосунок "Фудамент" - це веб-сервіс для класифікації зображень, заснований на використанні згорткових нейронних мереж (CNN) та оптимізований для роботи із датаcетоv [cifar-10](https://www.kaggle.com/c/cifar-10). . Проект дозволяє користувачам завантажувати зображення, які класифікуються за допомогою попередньо тренованих моделей, таких VGG16. Сервіс підтримує широкий спектр функцій, включаючи завантаження зображень, тренування моделей, перевірку якості зображень, а також детальне відображення результатів класифікації. Особлива увага приділяється оптимізації моделей для зменшення їх розміру та підвищення точності класифікації, що робить сервіс ефективним інструментом для обробки зображень з cifar-10.

## <p align="center">:e-mail: Команда проекту "Фундамент"</p>
---

### <p align="center">Volodymyr Rizun :calling:[Github](https://github.com/VolodymyrRiz) :postbox: [Linkedin](https://www.linkedin.com/in/volodymyr-rizun-40103219b/)</p>

### <p align="center">Vladyslav Skopenko :calling:[Github](https://github.com/VladSkopenko) :postbox: [Linkedin](https://www.linkedin.com/in/vladyslav-skopenko/)</p>

### <p align="center">Viktoriia Kalachova :calling:[Github](https://github.com/ViKalachova) :postbox: [Linkedin](https://www.linkedin.com/in/viktoriia-kalachova-746228282/)</p> 

### <p align="center">Borys Kuchyn :calling:[Github](https://github.com/Zpfamily) :postbox: [Linkedin](https://www.linkedin.com/in/borys-kuchyn-1316581a4/)</p>

### <p align="center">Dmytro Ostrenko  :calling:[Github](https://github.com/Dmytro-Ostrenko) :postbox: [Linkedin](https://www.linkedin.com/in/dmytro-ostrenko/)</p>
---

## <p align="center">:open_file_folder: Технології, які були використані у проекті</p> 

- **[Python](https://www.python.org/)** - основна мова програмування для розробки логіки класифікації зображень та роботи з нейронними мережами.
- **[Django](https://www.djangoproject.com/)** - веб-фреймворк для створення та управління бекендом веб-додатку.
- **[TensorFlow](https://www.tensorflow.org/)** - бібліотека машинного навчання для створення, тренування та використання моделей нейронних мереж.
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** - платформа для контейнеризації, що забезпечує ізольоване середовище для розгортання додатку.
- **[Docker Hub](https://hub.docker.com/)** - регістр контейнерів для зберігання та розповсюдження Docker-образів.
- **[GitHub](https://github.com/)** - платформа для керування версіями коду та співпраці над проектом.
- **[GitHub Desktop](https://github.com/apps/desktop)** - інструмент для зручного керування репозиторіями GitHub на локальному комп'ютері.
- **[VSCode](https://code.visualstudio.com/) та [PyCharm](https://www.jetbrains.com/ru-ru/pycharm/)** - основні середовища розробки для роботи над кодом проекту.


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

### Контейнеризація за допомогою Docker

Щоб контейнеризувати програму в Docker і завантажити образ на Docker Hub, виконайте наступні кроки:

1. **Dockerfile** :whale::

`Dockerfile` необхіден створення Docker-образу, що дозволяє розміщувати і запускати програму в контейнеризованому середовищі. `Dockerfile` у `Data09/data9/` включає всі необхідні інструкції для створення образу, такі як вибір базового образу, копіювання вихідного коду програми до контейнера, встановлення необхідних залежностей та визначення команди для запуску програми.

2. **Інтеграція Docker Compose** :whale::whale::

Інструмент Docker Compose необхіден для спрощення процесу розгортання та управління нашим проєктом у середовищі Docker. Файл `docker-compose.yml` у `Data09/data9/`, який описує параметри, мережі та томи, необхідні для проєкту. Цей файл дозволяє запускати весь проєкт за допомогою однієї команди `docker-compose up`, автоматизуючи створення та запуск необхідних Docker-контейнерів.

3. **Збірка та завантаження Docker-образу на Docker Hub** :rocket::

Після налаштування `Dockerfile` та `docker-compose.yml` створимо Docker-образ і завантажимо його на Docker Hub для легкого доступу та розгортання:

    ```
    docker build -t User/fundament-app .
    docker login
    docker push User/fundament-app
    ```
Наразі у Docker Hub вже є Docker-образ, який можливо підняти наступними командами:

    ```
    docker pull goituau/data9-web:latest
    docker run -d -p 8000:8000 goituau/data9-web:latest
    ```
    




4. **Розгортання програми за допомогою Docker Compose** :white_check_mark::

    Нарешті, розгорніть програму за допомогою Docker Compose:

    ```
    docker-compose up
    ```

Ці кроки допоможуть вам налаштувати додаток "Фундамент" за допомогою Docker, що полегшить його розгортання та запуск у різних середовищах.


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

# Модель VGG16

Архітектура моделі, яка будується в цьому коді, базується на VGG16 з деякими модифікаціями для адаптації до задачі класифікації на датасеті CIFAR-10. Ось детальний опис архітектури:

- **Базова модель VGG16**: Використовується з попередньо навченими вагами на ImageNet, але без останнього повнозв'язного шару (`include_top=False`). Вхідний розмір зображення встановлений на (32, 32, 3).
- **GlobalAveragePooling2D**: Цей шар зменшує розмірність виходу з VGG16, обчислюючи середні значення по просторових вимірах.
- **Повнозв'язний шар (Dense layer)**: Кількість нейронів у цьому шарі визначається гіперпараметром `units`, який може приймати значення від 256 до 1024 з кроком 128. Функція активації — ReLU.
- **Dropout layer**: Використовується для регуляризації моделі, зменшуючи перенавчання. Швидкість dropout визначається гіперпараметром `dropout_rate`, який може приймати значення від 0.2 до 0.5 з кроком 0.1.
- **Останній повнозв'язний шар (Dense layer)**: Має 10 нейронів (відповідно до кількості класів у CIFAR-10) і функцію активації softmax для класифікації.

Модель компілюється з оптимізатором Adam, функцією втрат `sparse_categorical_crossentropy` (оскільки мітки є цілими числами, а не категоріальними векторами) і метрикою `accuracy`.

## Тюнінг гіперпараметрів

У коді використовується Keras Tuner для оптимізації гіперпараметрів моделі. Ось детальний опис того, як відбувається тюнінг гіперпараметрів:

### Функція побудови моделі з гіперпараметрами

- **`build_model(hp)`**: Ця функція приймає об'єкт `hp` від Keras Tuner і визначає гіперпараметри, які будуть оптимізовані.
  - **`units`**: Кількість нейронів у повнозв'язному шарі. Може приймати значення від 256 до 1024 з кроком 128.
  - **`dropout_rate`**: Швидкість dropout для регуляризації. Може приймати значення від 0.2 до 0.5 з кроком 0.1.
  - **`learning_rate`**: Швидкість навчання для оптимізатора Adam. Може приймати значення 1e-2, 1e-3 або 1e-4.

### Ініціалізація тюнера

- **`tuner = RandomSearch(...)`**: Використовується `RandomSearch` для пошуку найкращих гіперпараметрів. Встановлено максимальну кількість проб (`max_trials`) рівною 5.

### Пошук найкращих гіперпараметрів

- **`tuner.search(...)`**: Тюнер виконує пошук, тренуючи модель з різними комбінаціями гіперпараметрів. На кожній пробі модель тренується протягом 50 епох.

### Отримання найкращої моделі

- **`best_model = tuner.get_best_models(num_models=1)[0]`**: Після завершення пошуку тюнер вибирає найкращу модель на основі метрики `val_accuracy`.

### Навчання моделі з колбеками

- **`checkpoint` і `early_stopping`**: Використовуються для збереження найкращої моделі та запобігання перенавчанню відповідно.

### Збереження історії тренування

Історія тренування зберігається у форматі JSON для подальшої візуалізації.

Тюнінг гіперпараметрів у коді дозволив автоматично знаходити оптимальні значення для кількості нейронів у повнозв'язному шарі, швидкості dropout та швидкості навчання, що значно покращило продуктивність моделі на задачі класифікації CIFAR-10.

## Результати

![Модель VGG16](Data09/data9/image_analysis/static/vgg16.png)



### 2. AlexNet
[AlexNet](https://www.kaggle.com/code/blurredmachine/alexnet-architecture-a-complete-guide)  – це архітектура CNN, яка вперше продемонструвала силу глибокого навчання на конкурсі ImageNet у 2012 році. Модель складається з 8 навчальних шарів, включаючи 5 згорткових і 3 повнозв’язних. Особливості AlexNet включають використання великих фільтрів на початкових шарах та Dropout для запобігання перенавчанню.

### 3. LeNet
[LeNet](https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/) – одна з перших CNN архітектур, розроблена [Яном ЛеКуном](https://yann.lecun.com/) для розпізнавання рукописних цифр. Вона складається з чергування згорткових і підвибіркових (Pooling) шарів, завершуючись кількома повнозв’язними шарами. LeNet є простою в реалізації, але менш потужною порівняно з іншими сучасними архітектурами.

### 4. MobileNet
[MobileNet](https://keras.io/api/applications/mobilenet/)  – це легка архітектура CNN, спеціально розроблена для мобільних та вбудованих систем. Вона використовує глибокі згортки (Depthwise Separable Convolutions), що значно зменшує кількість параметрів і обчислень, роблячи її ефективною для застосувань з обмеженими ресурсами.


### <p align="center">:bulb: Опис фільтрів для роботи із зображеннями </p>

У програмі, за умови, що реальна точність буде меншою за точність, яку пропонується обрати користувачу,  застосовуються для додаткової обробки зображень фільтри. Якщо користувач не хоче використовувати фільтри достатньо встановити цей параметр на рівні «0».
Що ж до самих фільтрів, то вони мають наступні характеристики:


1. **[Контраст](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Contrast)**:
   - **Опис**: підвищує контрастність зображення, роблячи темніші області темнішими, а світліші області світлішими.
   - **Вплив**: підвищує різкість деталей та може допомогти виявити риси зображення, які інакше могли б бути непомітними.

2. **[Різкість](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Sharpness)**:
   - **Опис**: збільшує чіткість зображення, акцентуючи на краях та деталях.
   - **Вплив**: робить деталі зображення більш чіткими, що може допомогти в покращенні точності класифікації, якщо зображення було розмите або не чітке.

3. **[Колір](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Color)**:
   - **Опис**: підвищує інтенсивність кольорів на зображенні.
   - **Вплив**: робить кольори яскравішими та насиченішими, що може допомогти виявити кольорові деталі, які можуть бути важливими для класифікації.

4. **[Деталізація](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html)**:
   - **Опис**: збільшує рівень деталей у зображенні.
   - **Вплив**: Підвищує чіткість текстур і дрібних деталей, що може бути корисно для виявлення дрібних особливостей, які можуть бути важливими для моделі.

5. **[Додаткова різкість](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Sharpness)**:
   - **Опис**: додатково покращує різкість зображення, акцентуючи на краях.
   - **Вплив**: підкреслює контури та деталі, ніж стандартний фільтр різкості.

6. **[Згладжування](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html)**:
   - **Опис**: зменшує шум та незначні деталі, роблячи зображення м’якшим.
   - **Вплив**: може бути корисним для зменшення шуму, але може також призвести до розмивання деталей.

7. **[Розмиття](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.GaussianBlur)**:
   - **Опис**: розмиває зображення за допомогою гаусового розмиття.
   - **Вплив**: зменшує деталі і шум, що може бути корисно, якщо зображення є надто деталізованим або містить артефакти.

8. **[Накладання фонового кольору](https://pillow.readthedocs.io/en/stable/reference/Image.html)**:
   - **Опис**: перетворює зображення в відтінки сірого і накладає кольоровий фоновий ефект.
   - **Вплив**: зменшує кольорову інформацію та може допомогти виявити контури та структуру зображення, що може бути корисно в умовах недостатньої контрастності або кольорових відмінностей.

P.S. Ці фільтри використовуються для обробки зображень перед їх класифікацією, що може допомогти поліпшити точність прогнозів моделі у випадках, коли початкове зображення не зовсім підходить для безпосереднього аналізу.


## Використання

Проект "Фундамент" дозволяє користувачу вибрати одну з чотирьох моделей для класифікації зображень з датасету cifar-10. Попередньо натреновані моделі збережені  у форматі `.keras`, що дозволяє їх повторне використання без необхідності перенавчання.


____

## <p align="center">:cat: Language of reading - English. Main information about the project:cat: </p>

____
The "Fundament" application is a web service for image classification, based on the use of convolutional neural networks (CNN) and optimized for working with the [cifar-10](https://www.kaggle.com/c/cifar-10) dataset. The project allows users to upload images that are classified using pre-trained models, such as VGG16. The service supports a wide range of features, including image uploading, model training, image quality verification, and detailed display of classification results. Special attention is given to model optimization for reducing their size and increasing classification accuracy, making the service an effective tool for processing cifar-10 images.

## <p align="center">:e-mail: The "Fundament" Project Team</p>
---

### <p align="center">Volodymyr Rizun :calling:[Github](https://github.com/VolodymyrRiz) :postbox: [Linkedin](https://www.linkedin.com/in/volodymyr-rizun-40103219b/)</p>

### <p align="center">Vladyslav Skopenko :calling:[Github](https://github.com/VladSkopenko) :postbox: [Linkedin](https://www.linkedin.com/in/vladyslav-skopenko/)</p>

### <p align="center">Viktoriia Kalachova :calling:[Github](https://github.com/ViKalachova) :postbox: [Linkedin](https://www.linkedin.com/in/viktoriia-kalachova-746228282/)</p> 

### <p align="center">Borys Kuchyn :calling:[Github](https://github.com/Zpfamily) :postbox: [Linkedin](https://www.linkedin.com/in/borys-kuchyn-1316581a4/)</p>

### <p align="center">Dmytro Ostrenko  :calling:[Github](https://github.com/Dmytro-Ostrenko) :postbox: [Linkedin](https://www.linkedin.com/in/dmytro-ostrenko/)</p>
---

## <p align="center">:open_file_folder: Technologies used in the project</p> 

- **[Python](https://www.python.org/)** - The main programming language used for developing the image classification logic and working with neural networks.
- **[Django](https://www.djangoproject.com/)** - A web framework for building and managing the backend of the web application.
- **[TensorFlow](https://www.tensorflow.org/)** - A machine learning library for creating, training, and deploying neural network models.
- **[PostgreSQL](https://www.postgresql.org/)** - A relational database used for storing and managing user data and models.
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** - A platform for containerization that ensures an isolated environment for deploying the application.
- **[Docker Hub](https://hub.docker.com/)** - A container registry for storing and distributing Docker images.
- **[GitHub](https://github.com/)** - A platform for version control and collaboration on the project.
- **[GitHub Desktop](https://github.com/apps/desktop)** - A tool for convenient management of GitHub repositories on a local machine.
- **[VSCode](https://code.visualstudio.com/)** and **[PyCharm](https://www.jetbrains.com/pycharm/)** - The main development environments for working on the project's code.


## <p align="center">:scroll: Main features of the application</p>

1. **Image Upload for Classification**  
Users can upload images for classification, which are then processed by a neural network to determine the category.

2. **Interaction with Four Pre-trained Neural Networks**  
Users can select one of four neural network architectures and set a confidence threshold between 0 and 1. Depending on this level, different filters are applied to improve image detail.

3. **Upload and Analysis of Trained Neural Networks**  
Users can upload a trained neural network, view information about each network, and analyze training graphs on test and validation data.

4. **Containerization with Docker**  
The project is fully containerized using [Docker](https://www.docker.com/), which ensures ease of deployment and integration in various environments.

These features provide a convenient and flexible interface for working with neural networks and images, as well as simplifying the deployment of the application.

## <p align="center">:bookmark_tabs: Instructions for installation and use of the "Fundament" software</p>

### Prerequisites
Ensure that Python version 3.11 or higher is installed on your computer. You can download Python from the [official site](https://www.python.org/downloads/).

### Installation

Before you start using the "Fundament" web interface with artificial neural networks for image recognition, you need to install it. Follow these steps:

1. **Clone the repository to your computer** :white_check_mark::   

```
git clone https://github.com/Dmytro-Ostrenko/Data09
```

2. **Navigate to the project directory** :white_check_mark::  

```
cd Data09
```

3. **Install dependencies from `requirements.txt`** :white_check_mark::
   
```
pip install -r requirements.txt 
```

4. **Create and activate a virtual environment** :white_check_mark::
   
```
python -m venv venv
source venv/bin/activate  # for Linux/MacOS
.\venv\Scripts\activate   # for Windows
```
Currently, Docker Hub already has a Docker image that can be brought up with the following commands:

 ```
 docker pull goituau/data9-web:latest
 docker run -d -p 8000:8000 goituau/data9-web:latest
 ```


5. **Update dependencies (optional)** :white_check_mark::

```
pip install --upgrade -r requirements.txt
```

6. **Run the application** :white_check_mark::

```
cd Data09/data9
python manage.py runserver
```


You can also quickly [download](https://github.com/Dmytro-Ostrenko/Data09/archive/refs/heads/main.zip) or [clone](https://github.com/Dmytro-Ostrenko/Data09.git).

### Docker Containerization

To containerize the program in Docker and upload the image to Docker Hub, follow these steps:

1. **Dockerfile** :whale::

    The `Dockerfile` is required to create a Docker image, allowing the deployment and execution of the program in a containerized environment. The `Dockerfile` located in `Data09/data9/` includes all the necessary instructions for building the image, such as selecting a base image, copying the source code into the container, installing the required dependencies, and defining the command to run the application.

2. **Docker Compose Integration** :whale::whale::

    The Docker Compose tool is necessary to simplify the deployment and management of our project in a Docker environment. The `docker-compose.yml` file located in `Data09/data9/` describes the services, networks, and volumes needed for the project. This file allows running the entire project with a single `docker-compose up` command, automating the creation and launch of the required Docker containers.

3. **Build and Push the Docker Image to Docker Hub** :rocket::

    After configuring the `Dockerfile` and `docker-compose.yml`, we will build the Docker image and push it to Docker Hub for easy access and deployment:

    ```bash
    docker build -t User/fundament-app .
    docker login
    docker push User/fundament-app
    ```

4. **Deploy the Application with Docker Compose** :white_check_mark::

    Finally, deploy the application using Docker Compose:

    ```bash
    docker-compose up
    ```

These steps will help you set up the "Fundament" application with Docker, making it easier to deploy and run in different environments.

## <p align="center">:card_index: Types of Neural Networks Used in the Project</p>

### Overview

This project implements image classification on the CIFAR-10 dataset using four different neural network architectures. Below are the types of models used and their accuracy:

1. **VGG16** (accuracy 90%)
2. **AlexNet** (accuracy 89%)
3. **LeNet** (accuracy 79%)
4. **MobileNet** (accuracy 86%)

### Model Descriptions

### 1. VGG16
[VGG16](https://keras.io/api/applications/vgg/) is one of the most popular convolutional neural network (CNN) architectures, consisting of 16 layers. The main idea of VGG16 is to use small filters of size 3x3 in all convolutional layers, which preserves spatial information and achieves high accuracy. The model ends with several fully connected layers and a Softmax layer for classification.

### 2. AlexNet
[AlexNet](https://www.kaggle.com/code/blurredmachine/alexnet-architecture-a-complete-guide) is a CNN architecture that first demonstrated the power of deep learning at the ImageNet competition in 2012. The model consists of 8 learnable layers, including 5 convolutional and 3 fully connected layers. AlexNet's features include the use of large filters in the initial layers and Dropout to prevent overfitting.

### 3. LeNet
[LeNet](https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/) is one of the first CNN architectures, developed by [Yann LeCun](https://yann.lecun.com/) for recognizing handwritten digits. It consists of alternating convolutional and pooling layers, ending with several fully connected layers. LeNet is simple to implement but less powerful compared to other modern architectures.

### 4. MobileNet
[MobileNet](https://keras.io/api/applications/mobilenet/) is a lightweight CNN architecture specifically designed for mobile and embedded systems. It uses depthwise separable convolutions, which significantly reduces the number of parameters and computations, making it efficient for resource-constrained applications.

### <p align="center">:bulb: Description of Image Filters</p>

In the program, if the actual accuracy is lower than the accuracy the user is expected to choose, filters are applied for additional image processing. If the user does not want to use filters, it is enough to set this parameter to "0".
As for the filters themselves, they have the following characteristics:

1. **[Contrast](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Contrast)**:
   - **Description**: Increases the contrast of the image, making darker areas darker and lighter areas lighter.
   - **Impact**: Enhances the sharpness of details and can help reveal features of the image that might otherwise be unnoticed.

2. **[Sharpness](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Sharpness)**:
   - **Description**: Increases the clarity of the image by accentuating edges and details.
   - **Impact**: Makes the details of the image clearer, which can improve classification accuracy if the image was blurry or unclear.

3. **[Color](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Color)**:
   - **Description**: Enhances the intensity of colors in the image.
   - **Impact**: Makes colors more vivid and saturated, which can help reveal color details that may be important for classification.

4. **[Detail](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html)**:
   - **Description**: Increases the level of detail in the image.
   - **Impact**: Enhances the clarity of textures and fine details, which can be useful for detecting small features that might be important for the model.

5. **[Additional Sharpness](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Sharpness)**:
   - **Description**: Further improves the sharpness of the image by accentuating edges.
   - **Impact**: Emphasizes contours and details more than the standard sharpness filter.

6. **[Smoothing](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html)**:
   - **Description**: Reduces noise and minor details, making the image smoother.
   - **Impact**: Can be useful for reducing noise but may also lead to blurring of details.

7. **[Blur](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.GaussianBlur)**:
   - **Description**: Blurs the image using Gaussian blur.
   - **Impact**: Reduces details and noise, which can be useful if the image is overly detailed or contains artifacts.

8. **[Color Background Overlay](https://pillow.readthedocs.io/en/stable/reference/Image.html)**:
   - **Description**: Converts the image to grayscale and applies a color background effect.
   - **Impact**: Reduces color information and can help reveal the contours and structure of the image, which can be useful in cases of low contrast or color differences.

P.S. These filters are used for preprocessing images before classification, which can help improve the accuracy of the model’s predictions in cases where the initial image is not well-suited for direct analysis.

## Usage

The "Fundament" project allows users to choose one of four models for classifying images from the CIFAR-10 dataset. Pre-trained models are saved in `.keras` format, which allows for their reuse without the need for retraining.






