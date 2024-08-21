#  <p align="center">:robot:  Data09:robot:  </p> 

> <p align="center">Implementation of a web service that classifies images using convolutional neural networks</p> 

## <p align="center">:smiley_cat: Мова читання - українська. Основна інформація про проект</p>

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
- **[PostgreSQL](https://www.postgresql.org/)** - реляційна база даних для збереження та управління даними користувачів та моделями.
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

## Usage

The "Fundament" project allows users to choose one of four models for classifying images from the CIFAR-10 dataset. Pre-trained models are saved in `.keras` format, which allows for their reuse without the need for retraining.






