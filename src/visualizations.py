from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text):
    """Generates a WordCloud figure from the provided text."""
    if not text or len(text.strip()) == 0:
        return None

    # Create word cloud object
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='#161b22',
        colormap='cool',
        max_words=100
    ).generate(text)

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#161b22')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    fig.tight_layout(pad=0)
    
    return fig
