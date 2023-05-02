## Steps to run the app:

1. Create and activate a python virtual environment.
```
python3 -m venv env
```
and then activate the environment,
```
source env/bin/activate
```

2. Install requirements.txt.
```
pip install -r requirements.txt
```

3. Apply migrations.
```
python manage.py migrate
```

4. Run the app.
```
python manage.py runserver
```
## Screenshots:
### Initial screen:
![Screenshot from 2023-05-03 00-55-54](https://user-images.githubusercontent.com/52523023/235769122-ab738268-9545-41bc-ac68-7d9d2bffb274.png)

### Upload .wav/.mp3 file:
![Screenshot from 2023-05-03 01-13-45](https://user-images.githubusercontent.com/52523023/235769307-8fdbdb9c-63c9-42d0-82c5-e52c08b35d8c.png)

### Prediction:
![Screenshot from 2023-05-03 00-55-41](https://user-images.githubusercontent.com/52523023/235769331-8be87b64-72bc-4533-bbb1-94466007343a.png)
