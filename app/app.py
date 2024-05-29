from auth import ForgotPasswordForm, auth_blueprint,  email_us, login_user, logout_user, register_user, review_us, subscribeus, updatedpassword
from flask import Flask, redirect, render_template, abort, session, url_for
from flask_mysqldb import MySQL
from flask_mail import Mail
from itsdangerous import URLSafeTimedSerializer
from flask_mysqldb import MySQL
from flask_mysqldb import MySQL
from flask_mail import Mail

app = Flask(__name__, template_folder="../frontend",
            static_folder="../frontend")

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'fyp.se18@gmail.com'
app.config['MAIL_PASSWORD'] = '-'
mail = Mail(app)

app.secret_key = '12345678'
serializer = URLSafeTimedSerializer(app.secret_key)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '12345678'
app.config['MYSQL_DB'] = 'fyp'
app.config['MYSQL_PORT'] = 3306
mysql = MySQL(app)

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'adminadmin'

app.register_blueprint(auth_blueprint, url_prefix='/auth')



# ---------------------------------------------------AUTHENTICATION
@app.route('/login', methods=['GET', 'POST'])
def login():
    return login_user(mysql)


@app.route('/register', methods=['GET', 'POST'])
def register():
    return register_user(mysql)


@app.route('/logout')
def logout():
    return logout_user()


@app.route('/userlogin')
def userlogin():
    return render_template('userlogin.html')

# --------------------------------------------PASSWORD, EMAIL, NEWSLETTER, REVIEW


@app.route('/emailus', methods=['GET', 'POST'])
def emailus():
    return email_us(mysql)


@app.route('/subscribe', methods=['GET', 'POST'])
def subscribe():
    return subscribeus(mysql)


@app.route('/forgotpass', methods=['GET', 'POST'])
def forgotpass():
    form = ForgotPasswordForm()
    return render_template('forgotpass.html', form=form)


@app.route('/updatepassword', methods=['POST'])
def updatepassword():
    return updatedpassword(mysql)


@app.route('/newsletter')
def newsletter():
    return render_template('newsletter.html')


@app.route('/review')
def review():
    return render_template('review.html')


@app.route('/reviewus', methods=['GET', 'POST'])
def reviewus():
    return review_us(mysql)


@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
