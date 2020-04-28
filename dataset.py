from csv import DictReader
import os
import re


class DataSet:
    def __init__(self, name="train", path="fnc-1"):
        self.path = path

        print("Reading dataset")
        bodies = name+"_bodies.csv"
        stances = name+"_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        # make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])
            s['Headline'] = DataSet.clean_article(s['Headline'])

        # copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = DataSet.clean_article(article['articleBody'])

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))

    @staticmethod
    def clean_article(article):
        return re.sub(r'\s+', ' ', article).encode('ascii', 'ignore').decode('utf-8')

    def read(self, filename):
        rows = []
        with open(os.path.join(self.path, filename), "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
