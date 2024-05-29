import os
import sys
import unittest
from unittest.mock import Mock, patch
import MySQLdb
import MySQLdb.cursors
from flask import Flask, session
from flask.testing import FlaskClient
import pandas as pd

from auth import process_and_suggest_professors, process_and_suggest, register_user, logout_user, login_user

class BBtest1(unittest.TestCase):
    def test_topics(self):
        
        print("\nBlack box testing 2: For suggesting trending topics based on abstract")

        abstract = "This is a sample abstract containing keywords related to the topic"
        trending_topics = process_and_suggest(abstract)

        # case if the result is a list
        self.assertIsInstance(trending_topics, list)
        print("\ncase 1: data available passed.")

        # case if the list is not empty
        self.assertTrue(trending_topics)
        print("\ncase 2: check empty passed.") 

class BBtest2(unittest.TestCase):
    def test_professors(self):
        print("\nTESTING\nBlack box testing 1: For suggesting professors based on abstract")
        
        abstract = "This is a sample abstract containing keywords related to the topic"
        pred_prof = process_and_suggest_professors(abstract)

        # 1: data available -------------------------------------
        self.assertIsInstance(pred_prof, pd.DataFrame)
        print("\ncase 1: check passed.")
        print("\ncase 1: check passed.")

        # 2 if it hass dame columns -----------------------------------
        expected_columns = ['Name', 'University Name', 'Email', 'Research Interests', 'design', 'Country']
        self.assertListEqual(list(pred_prof.columns), expected_columns)
        print("\ncase 2:  check cols passed.")
        print("\ncase 2:  check cols passed.")

        # if its not empty -------------------------------------
        self.assertFalse(pred_prof.empty)
        print("\ncase 3: check empty passed")
        print("\ncase 3: check empty passed")

        #rough flase test------------------------------------
        #self.assertTrue(False) 
        #print("case 4: flase test passed") 


class WBtest1(unittest.TestCase):
    def test_professors(self):
        print("\nTESTING\nWhite Box box testing 1: For suggesting professors based on abstract")
        # Test for a sample abstract with known keywords
        keyword = "This is an abstract about cloud computing security"
        expected_professors = [('John Doe', 'University A', 'john.doe@example.com', 'Cloud Computing Security', 'Professor', 'USA')]
        actual_professors = process_and_suggest_professors(keyword)
        self.assertEqual(actual_professors, expected_professors)
        print("WBT1: test passed (actual professors and expected professors are equal)") 

class WBtest2(unittest.TestCase):
    def test_topics(self):
            print("\nTESTING\nBlack box testing 2: For suggesting topics based on abstract")
            # Test for a sample abstract with known keywords
            keyword = "This is an abstract about internet of things"
            expected_topics = ['Internet of Things', 'IoT Security', 'IoT authlications']
            actual_topics = process_and_suggest(keyword)
            self.assertListEqual(actual_topics, expected_topics)
            print("WBT2: test passed (actual topics and expected topics are equal)") 


            

if __name__ == '__main__':
    unittest.main()
 
