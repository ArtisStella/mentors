import os
import re
import torch
import random
import pandas as pd
import MySQLdb.cursors
from mailbox import Message
from flask_wtf import FlaskForm
from flask_mail import Mail, Message
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from data import dataset, trendingtopics
from transformers import BertTokenizer, BertModel
from wtforms.validators import InputRequired, Length
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from wtforms import Form, StringField, PasswordField, SubmitField
from flask import Blueprint, redirect, render_template, request, session, url_for


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


def preprocess(text):
    try:
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = word_tokenize(text.lower())
        clean_tokens = [stemmer.stem(
            token) for token in tokens if token.isalnum() and token not in stop_words]
        return ' '.join(clean_tokens)
    except Exception as e:
        print("Error preprocessing text (user's abstract): ", e)
        return None


def load_bert_model_and_tokenizer():
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer_path = os.path.join(cache_dir, "bert-base-uncased-tokenizer")
    model_path = os.path.join(cache_dir, "bert-base-uncased-model")
    if os.path.exists(tokenizer_path) and os.path.exists(model_path):
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        model = BertModel.from_pretrained(model_path)
        print("BERT model and tokenizer loaded from cache.")
    else:
        print("BERT model and tokenizer not found in cache. Downloading...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(tokenizer_path)
        model.save_pretrained(model_path)
        print("BERT model and tokenizer downloaded and saved to cache.")
    return tokenizer, model


def encode_texts(texts, tokenizer, model, batch_size=32):
    num_texts = len(texts)
    encoded_embeddings = []
    for i in range(0, num_texts, batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        batch_embeddings = model_output.last_hidden_state.mean(dim=1)
        encoded_embeddings.append(batch_embeddings)
    return torch.cat(encoded_embeddings)


def process_text(text):
    try:
        print("Input is: ", text)
        preprocessed_input = preprocess(text)
        print("Preprocessed input is: ", preprocessed_input)

        if preprocessed_input is None:
            return "Input text could not be processed. Please try again."
        abstracts = [abstract for abstract, domain in dataset]
        preprocessed_abstracts = [preprocess(
            abstract) for abstract in abstracts]

        print("Preprocessed abstracts: ", preprocessed_abstracts)
        combined_texts = preprocessed_abstracts + [preprocessed_input]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(combined_texts)

        print("TF-IDF matrix shape: ", tfidf_matrix.shape)
        tfidf_similarity_scores = cosine_similarity(
            tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        print("TF-IDF Similarity scores: ", tfidf_similarity_scores)
        tokenizer, model = load_bert_model_and_tokenizer()
        bert_embeddings = encode_texts(combined_texts, tokenizer, model)
        bert_similarity_scores = cosine_similarity(
            bert_embeddings[-1].numpy().reshape(1, -1), bert_embeddings[:-1].numpy()).flatten()
        print("BERT Similarity scores: ", bert_similarity_scores)
        combined_similarity_scores = (
            tfidf_similarity_scores + bert_similarity_scores) / 2
        similar_abstracts = [(abstract, domain, similarity) for (
            abstract, domain), similarity in zip(dataset, combined_similarity_scores)]
        similar_abstracts = [
            item for item in similar_abstracts if item[2] > 0.0]
        similar_abstracts.sort(key=lambda x: x[2], reverse=True)
        if not similar_abstracts:
            return "No similar abstracts found. Please try another search term."
        top_similar = similar_abstracts[:10]
        prediction = top_similar[0][1]
        print("Suggested prediction:", prediction)
        return prediction
    except Exception as e:
        print("Error suggesting prediction:", e)
        return None


def load_trending_topics():
    trending_topics = [(topic, preprocess(topic)) for topic in trendingtopics]
    return trending_topics


def calculate_similarity(text1, text2):
    try:
        # print("vectorizing trending topics. \nCalculating Similarity scores for trending topics")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([text2, text1])
        similarity_score = (X * X.T).A[0, 1]
        return similarity_score
    except Exception as e:
        print("Error calculating trending topics similarity: ", e)
        return None


def suggest_trending_topics(predicted_topic, trending_topics):
    try:
        similar_topics = []
        for topic, topic_text in trending_topics:
            similarity = calculate_similarity(predicted_topic, topic_text)
            if similarity > 0.0:
                similar_topics.append((topic, similarity))
        similar_topics.sort(key=lambda x: x[1], reverse=True)
        similar_topics.sort(key=lambda x: x[1], reverse=True)
        print("calculating similar topics....similar topics are",
              similar_topics[:3])
        top_similar = similar_topics[:10]
        return top_similar
        return top_similar
    except Exception as e:
        print("Error suggesting trending topics: ", e)
        return None


def process_and_suggest(predicted_topic):
    try:
        trending_topics = load_trending_topics()
        if trending_topics is None:
            return None

        if not trending_topics:
            print("No trending topics found.")
            return None

        trending_topics = suggest_trending_topics(
            predicted_topic, trending_topics)
        print("\nTrending topics are: ", trending_topics[:3])
        return trending_topics
    except Exception as e:
        print("Error processing and suggesting trending topics: ", e)
        return None


def load_professors_data():
    try:
        file_path = '/home/ubuntu/mentors/app/professors_data.xlsx'
        professors_data = pd.read_excel(file_path)
        return professors_data
    except FileNotFoundError:
        print("Error: professors_data.xlsx file not found.")
        return None


professors_data = load_professors_data()


def calculate_professors_similarity(predicted_topic):
    try:
        # Preprocess predicted topic
        predicted_topic_processed = preprocess(predicted_topic)
        if predicted_topic_processed is None:
            raise ValueError("Error: Preprocessed predicted topic is None")

        # Check if professors data is loaded
        if professors_data is None:
            raise ValueError("Error: Professors' data is not loaded")

        # Preprocess research interests
        research_interests = professors_data['Research Interests'].fillna(
            '').str.lower().str.replace('[^a-zA-Z\s]', '')
        if research_interests.empty:
            raise ValueError("Error: Research interests column is empty")

        # Vectorize research interests
        vectorizer = TfidfVectorizer()
        research_interests_tfidf = vectorizer.fit_transform(research_interests)
        if research_interests_tfidf.shape[0] == 0:
            raise ValueError(
                "Error: TF-IDF vectorization resulted in an empty matrix")

        # Vectorize predicted topic
        predicted_topic_tfidf = vectorizer.transform(
            [predicted_topic_processed])
        if predicted_topic_tfidf.shape[0] == 0:
            raise ValueError(
                "Error: TF-IDF vectorization for predicted topic resulted in an empty matrix")

        # Calculate similarity scores
        similarity_scores = cosine_similarity(
            predicted_topic_tfidf, research_interests_tfidf)
        if similarity_scores.size == 0:
            raise ValueError("Error: Similarity scores array is empty")

        # Sort professors indices by similarity score
        sorted_professors_indices = similarity_scores.argsort()[0][::-1]
        top_n = 5
        top_professors = professors_data.iloc[sorted_professors_indices[:top_n]]
        print("Top 5 similar professors saved")
        return top_professors

    except Exception as e:
        print("Error calculating professors similarity:", e)
        return None


def process_and_suggest_professors(predicted_topic):
    try:
        print("Suggesting professors......")
        # predicted_topic = process_text(text)
        related_professors = calculate_professors_similarity(predicted_topic)
        print("predicted keyword: ", predicted_topic,
              "\ntop 5 related professors: ", related_professors)
        print("predicted keyword: ", predicted_topic,
              "\ntop 5 related professors: ", related_professors)
        return related_professors
    except Exception as e:
        print("Error processing and suggesting professors: ", e)
        return None


def fetch_matching_trending_topics(user_query, mysql):
    query_fetch_matching_trending_topics = "SELECT * FROM trending_topics WHERE Lower(TopicKeyword) LIKE '%" + user_query.lower(
    ) + "%'"
    placeholders = '' + user_query.lower() + ''
    return execute_query(query_fetch_matching_trending_topics, None, mysql)
