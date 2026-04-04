import matplotlib.pyplot as plt
import seaborn as sns


def plot_content_type(df):

    fig, ax = plt.subplots()

    sns.countplot(data=df, x="type", ax=ax)

    ax.set_title("Movies vs TV Shows")

    return fig


def plot_release_trend(df):

    year_counts = df['release_year'].value_counts().sort_index()

    fig, ax = plt.subplots()

    ax.plot(year_counts.index, year_counts.values)

    ax.set_title("Content Release Trend")

    return fig


def plot_rating_distribution(df):

    fig, ax = plt.subplots()

    sns.countplot(data=df, y="rating", order=df['rating'].value_counts().index)

    ax.set_title("Ratings Distribution")

    return fig


def plot_country_distribution(df):

    top_countries = df['country'].value_counts().head(10)

    fig, ax = plt.subplots()

    sns.barplot(x=top_countries.values, y=top_countries.index)

    ax.set_title("Top Countries Producing Content")

    return fig