# Alwayswannacode/Цифровой прорыв project
# Python version: 3.11.9

*If you use windows, please set \ instead of / in the paths*

**Install module for venv-creating:**\
pip install venv

**Create venv:**\
python3 -m venv venv

**venv** - name of your virtual environment (last name in the string)

**activate your venv:**
**further in the instructions I will use "venv" instead of the name of my virtual environment**\
_linux:_

- source venv/Scripts/activate\
_PowerShell:_

- . venv\Scripts\activate.bat

**if you havent Scripts folder:**

- source venv/bin/activate
- . venv\bin\activate

**install all of the dependencies:**\
- pip install -r requirements.txt

**Env variables:**

- cp .env.example .env
**change some variables in .env file if you need it**
**in env you can set num of epochs for model fit and other things**

**Go into project directory:**

- cd sila

**Make migrations:**

- python3 manage.py migrate

**Collect static:**

- python3 manage.py collectstatic

**creating superuser:**

- python manage.py createsuperuser
- follow instructions

**Starting project:**

- go to sila folder for starting project (if you didnt do it on the previous step)
- cd sila

**Run server:**

- python3 manage.py runserver

**Deactivating venv:**

- deactivate
