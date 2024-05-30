import requests
from flask_mail import Mail
from flask_mysqldb import MySQL
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request, redirect, url_for, session
from auth import ForgotPasswordForm, auth_blueprint, login_user, register_user, email_us, logout_user, review_us, subscribeus, updatedpassword, fetch_reviews, fetch_matching_trending_topics, process_text, process_and_suggest, process_and_suggest_professors
from multiprocessing import process
from fuzzywuzzy import process


app = Flask(__name__, template_folder="../frontend",
            static_folder="../frontend")

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'fyp.se18@gmail.com'
app.config['MAIL_PASSWORD'] = '-'
mail = Mail(app)

app.secret_key = '12345678'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'flaskuser'
app.config['MYSQL_PASSWORD'] = 'Password123!'
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


@app.route('/logout')
def logout():
    return logout_user()


@app.route('/userlogin')
def userlogin():
    return render_template('userlogin.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    return register_user(mysql)
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
    reviews = fetch_reviews(mysql)
    message = request.args.get('message')
    # msg =request.args.get('msg')
    review1 = reviews[0] if len(reviews) > 0 else None
    review2 = reviews[1] if len(reviews) > 1 else None
    review3 = reviews[2] if len(reviews) > 2 else None
    review4 = reviews[3] if len(reviews) > 3 else None
    review5 = reviews[4] if len(reviews) > 4 else None
    review6 = reviews[5] if len(reviews) > 5 else None
    cards = [
        {'title': 'IOT', 'query': 'iot'},
        {'title': 'Machine Learning', 'query': 'machine+learning'},
        {'title': 'Digital Forensics', 'query': 'digital+forensics'},
        {'title': 'Blockchain', 'query': 'blockchain'},
        {'title': 'Vehicular Ad Hoc Networks',
            'query': 'vehicular+ad+hoc+networks'},
        {'title': 'Wireless Sensor Networks', 'query': 'wireless+sensor+networks'},
        {'title': 'Cloud Computing', 'query': 'cloud+computing'},
        {'title': 'Fog Computing', 'query': 'fog+computing'},
        {'title': 'Edge Computing', 'query': 'edge+computing'},
        {'title': 'Cloud Security', 'query': 'cloud+security'},
        {'title': 'Mobile Cloud Computing (MCC)',
         'query': 'mobile+cloud+computing'},
        {'title': 'Data Mining', 'query': 'data+mining'},
        {'title': 'Big Data', 'query': 'big+data'},
        {'title': 'Web Technology', 'query': 'web+technology'},
        {'title': 'Mobile Ad Hoc Networks (MANET)',
         'query': 'mobile+ad+hoc+networks'},
    ]

    googleuser = 'google_user' in session
    loggedin = 'loggedin' in session
    useronline = 'google_id' in session
    return render_template('home.html', r1=review1, r2=review2, r3=review3, r4=review4, r5=review5, r6=review6,  message=message, googleuser=googleuser, loggedin=loggedin, useronline=useronline, cards=cards)
# -------------------------------------------------------------------------SEARCH BY ASBTRACT


@app.route('/abstract')
def abstract():
    return render_template('abstract.html')


@app.route('/abstractresult', methods=['POST'])
def extract_keywords():
    text = request.form.get('text')
    predicted_topic = process_text(text)
    trending_topics = process_and_suggest(predicted_topic)
    related_professors = process_and_suggest_professors(predicted_topic)
    # dataframe mai can't use split n ek string mai display karrha loc se bhi nhi horha, isliye dict har value alag no \n ot \t prob also
    myls = related_professors.to_dict(orient='records')

    return render_template('abstract_result.html', keywords=predicted_topic, trending_topics=trending_topics, related_professors=related_professors, myls=myls)

# ----------------------------------------------------------RESEARCH PAPERS AND TRENDING TOPICS


@app.route('/researchpapers')
def researchpapers():
    if 'id' in session:
        return render_template('ResearchPapers.html')
    else:
        return render_template('home.html', msg="please login to access this page")


@app.route('/trending')
def trending():
    return render_template('trending.html')

# ------------------------------------------------SEARCH BY DOMAIN


@app.route('/search')
def search():
    userSearch = request.args.get('query')
    search_results = []

    try:
        cursor = mysql.connection.cursor()
        query = "SELECT mentor_id, Name, UniversityName, Email, ResearchInterests, Designation, Country FROM mentors"
        cursor.execute(query)
        data_from_database = cursor.fetchall()

        research_interests = [row[4] for row in data_from_database]

        matched_interests = process.extract(
            userSearch, research_interests, limit=10)

        matched_rows = [row for row in data_from_database if row[4] in [
            match[0] for match in matched_interests]]

        text_data = [row[1] + ' ' + row[2] for row in matched_rows]

        if text_data:
            vector_for_conversion = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vector_for_conversion.fit_transform(text_data)

            knn_model = NearestNeighbors(n_neighbors=len(
                matched_rows), algorithm='brute', metric='cosine')
            knn_model.fit(tfidf_matrix)

            user_input_vector = vector_for_conversion.transform([userSearch])

            indices = knn_model.kneighbors(
                user_input_vector, return_distance=False)[0]

            for index in indices:
                nearest_neighbor_data = matched_rows[index]
                search_results.append(nearest_neighbor_data)

    except Exception as e:
        print("Error connecting to MySQL database:", e)
    finally:
        cursor.close()

    return render_template('search_results.html', userQuery=userSearch, results=search_results)


@app.route('/trending_topics')
def trending_topics():
    user_query = request.args.get('query', '')
    matching_trending_topics = fetch_matching_trending_topics(
        user_query, mysql)
    if not matching_trending_topics:
        return render_template('trending_topics.html', no_topics=True)
    else:
        return render_template('trending_topics.html', matching_topics=matching_trending_topics)


@app.route('/profile')
def profile():
    name = request.args.get('name')
    universityname = request.args.get('universityname')
    email = request.args.get('email')
    researchinterest = request.args.get('researchinterest')
    designation = request.args.get('designation')
    country = request.args.get('country')
    return render_template('profile.html', name=name, universityname=universityname, email=email,
                           researchinterest=researchinterest, designation=designation, country=country)

# -----------------------------------------------ADMIN------------------------------------------------


@app.route('/adminpanel')
def adminpanel():
    print("Admin panel function executed!")
    return render_template('admin/loginadmin.html')


@app.route('/adminhome', methods=['POST'])
def adminhome():
    adminusername = request.form.get('adminusername')
    adminpassword = request.form.get('adminpassword')

    if adminusername == ADMIN_USERNAME and adminpassword == ADMIN_PASSWORD:
        session['adminusername'] = adminusername
        return redirect(url_for('admindashboard'))
    else:
        return render_template('admin/loginadmin.html', message='Invalid username or password')


@app.route('/admindashboard')
def admindashboard():
    if 'adminusername' in session:
        return render_template('admin/dashboardadmin.html')
    else:
        return redirect(url_for('admin/adminlogin'))


@app.route('/adminlogout')
def adminlogout():
    session.pop('adminusername', None)
    return render_template('admin/loginadmin.html')


@app.route('/adminprofessors')
def adminprofessors():
    conn = mysql.connection
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mentors")
    mentors = cursor.fetchall()
    return render_template('admin/Professors.html', mentors=mentors)


@app.route('/adminuniversities')
def adminuniversities():
    conn = mysql.connection
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT UniversityName FROM mentors")
    UniversityName = cursor.fetchall()
    return render_template('admin/Universities.html', UniversityName=UniversityName)


@app.route('/adminreviews')
def adminreviews():
    column_data = fetch_reviews(mysql)
    return render_template('admin/Reviews.html', column_data=column_data)


@app.route('/admintrending')
def admintrending():
    conn = mysql.connection
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trending_topics")
    topics = cursor.fetchall()
    return render_template('admin/Trending.html', topics=topics)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
