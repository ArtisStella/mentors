from mailbox import Message
import random
import re
from flask import Blueprint, redirect, render_template, request, session, url_for
from wtforms import Form, StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length
import MySQLdb.cursors
from flask_wtf import FlaskForm
from flask_mail import Mail, Message

auth_blueprint = Blueprint('auth', __name__)

# ------------------------------------------------FORGOT PASSWORD
class ForgotPasswordForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    oldpassword = PasswordField('Old Password', validators=[InputRequired()])
    newpassword = PasswordField('New Password', validators=[InputRequired()])
    submit = SubmitField('Change Password')


def updatedpassword(mysql):
    form = ForgotPasswordForm(request.form)
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'oldpassword' in request.form and 'newpassword' in request.form:
        username = request.form['username']
        oldpassword = request.form['oldpassword']
        newpassword = request.form['newpassword']
        if oldpassword == newpassword:
            msg = "New password cannot be same as the old password."
            return render_template('forgotpass.html', form=form, msg=msg)
        if not re.match(r'^(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{8,25}$', newpassword):
            msg = "Password must contain at least one capital letter, one numerical value, and be 8-25 characters long."
            return render_template('forgotpass.html', form=form, msg=msg)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM users WHERE username = %s AND password = %s', (username, oldpassword,))
        account = cursor.fetchone()
        if account:
            cursor.execute(
                'UPDATE users SET password = %s WHERE username = %s', (newpassword, username,))
            mysql.connection.commit()
            msg = "Password updated successfully! \n Login now"
            return render_template("userlogin.html", msg=msg)
        else:
            msg = "Incorrect username or old password!"
            return render_template('forgotpass.html', form=form, msg=msg)
    return render_template('forgotpass.html', form=form, msg=msg)
# --------------------------------------------------- Login


class LoginForm(Form):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    submit = SubmitField('Login')


def login_user(mysql):
    form = LoginForm(request.form)
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM users WHERE username = %s AND password =%s', (username, password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            session['password'] = account['password']
            msg = "Logged in successfully!"
            return render_template("home.html", msg=msg)
        else:
            msg = "Incorrect Credentials!"
    return render_template('userlogin.html',  msg=msg, form=form)

# --------------------------------------------- Register


class RegisterForm(Form):
    username = StringField('Username', validators=[
                           InputRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[
                             InputRequired(), Length(min=8, max=25)])
    email = StringField('Email', validators=[
                        InputRequired(), Length(min=9, max=25)])
    submit = SubmitField('Register')


def register_user(mysql):
    form = RegisterForm(request.form)
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists, Please login!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not re.match(r'^(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{8,25}$', password):
            msg = 'Password must contain at least one capital letter, one numerical value, and be 8-25 characters long.'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            cursor.execute(
                'INSERT INTO users (username, password, email) VALUES ( %s, %s, %s)', (username, password, email, ))
            mysql.connection.commit()
            msg = 'Registered successfully, Login Now!'
    elif request.method == 'POST':
        msg = 'Please fill out the form!'
    return render_template('userlogin.html', msg=msg, form=form)
# `--------------------------------------------- Logout


def logout_user():
    if 'google_id' in session:
        session.pop('google_id', None)
        session.pop('name', None)
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('home'))
# --------------------------------------------- Email us


class emailusform(Form):
    email = StringField('Email', validators=[
                        InputRequired(), Length(min=4, max=25)])
    usermsg = StringField('Text', validators=[
                          InputRequired(), Length(min=1, max=250)])
    # submit = SubmitField('Send Email')


def email_us(mysql):
    form = emailusform(request.form)
    msg = ''
    if request.method == 'POST' and form.validate():
        email = form.email.data
        usermsg = form.usermsg.data
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT id FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        if not user:
            msg = 'Please register yourself first!'
        else:
            user_id = user['id']
            cursor.execute(
                'INSERT INTO messages (user_id, email_from, message) VALUES (%s, %s, %s)', (user_id, email, usermsg))
            mysql.connection.commit()
            msg = 'Your email has been sent successfully!'
    elif request.method == 'POST':
        msg = 'Please fill out the form!'
    return render_template('home.html', form=form, msg=msg)
# --------------------------------------------------------------------------------- Review us


class ReviewForm(Form):
    username = StringField('Username', validators=[
                           InputRequired(), Length(min=1, max=25)])
    review = StringField('Review', validators=[
                         InputRequired(), Length(min=1, max=250)])
    email = StringField('Email', validators=[
                        InputRequired(), Length(min=4, max=25)])
    # submit = SubmitField('Send review')


def review_us(mysql):
    form = ReviewForm(request.form)
    msg = ''

    if request.method == 'POST':
        if 'username' in request.form and 'review' in request.form:
            username = form.username.data
            review = form.review.data
            if not username or not review:
                msg = 'Please fill out all the fields!'
            else:
                user_id = session.get('id')
                if user_id:
                    cursor = mysql.connection.cursor(
                        MySQLdb.cursors.DictCursor)
                    cursor.execute(
                        'INSERT INTO reviews (userid, username, review) VALUES (%s, %s, %s)', (user_id, username, review))
                    mysql.connection.commit()
                    msg = 'Your review has been sent successfully!'
                    return render_template("home.html", msg=msg)
                else:
                    msg = 'Please register yourself first!'
                    return render_template("home.html", msg=msg)
        else:
            msg = 'Please fill out the form!'
    return render_template('review.html', msg=msg, form=form)


class Review:
    def __init__(self, username, review):
        self.username = username
        self.review = review


def fetch_reviews(mysql):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT username, review FROM reviews')
    reviews_data = cursor.fetchall()
    reviews = [Review(review['username'], review['review'])
               for review in reviews_data]
    random.shuffle(reviews)  # Shuffle the list of reviews
    return reviews


# ---------------------------------------------------------------------------- Newsletter Subscription
mail = Mail()


def send_email(email):
    msg = Message('Thank You for Subscribing')
    msg.body = "Thank you for subscribing to our newsletter!"
    msg.recipients = [email]
    msg.sender = 'sumaiya.mmaqsood@gmail.com'
    mail.send(msg)


class subscribeusform(Form):
    email = StringField('Email', validators=[
                        InputRequired(), Length(min=4, max=25)])


def subscribeus(mysql):
    form = subscribeusform(request.form)
    msg = ''
    if request.method == 'POST' and 'email' in request.form:
        email = form.email.data
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT email FROM subscribers WHERE email = %s', (email,))
        user = cursor.fetchone()
        if user:
            msg = 'You have already subscribed!'
            return render_template('home.html', msg=msg)
            msg = 'You have already subscribed!'
            return render_template('home.html', msg=msg)
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        else:
            cursor.execute(
                'INSERT INTO subscribers (email) VALUES (%s)', (email,))
            mysql.connection.commit()
            # send_email(email)
            msg = 'You have been subscribed successfully!'
            return render_template('home.html', msg=msg)

    elif request.method == 'POST':
        msg = 'Please enter a valid email address!'
    return render_template('home.html', form=form, msg=msg)

# --------------------------------------------------------------------TRENDING TOPIC EXTRACTION BY KEYWORD


def execute_query(query, placeholders=None, mysql=None):
    print(query, placeholders)
    try:
        conn = mysql.connection
        cursor = conn.cursor()
        if placeholders:
            cursor.execute(query, placeholders)
        else:
            cursor.execute(query)
        if cursor.description:
            result = cursor.fetchall()
        else:
            result = None
        conn.commit()
        cursor.close()
        return result
    except Exception as e:
        print("An error occurred:", e)
        return None

# ------------------------------------------TRENDING TOPICS BY DOMAIN
