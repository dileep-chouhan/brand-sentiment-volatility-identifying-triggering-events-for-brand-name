import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_days = 100
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(num_days)]
brand_name = "[Brand Name]"
sentiment_scores = np.random.uniform(-1, 1, num_days) # -1: negative, 1: positive
events = ['No Event'] * num_days
# Inject some significant events
events[20] = 'Product Launch'
events[40] = 'Negative PR'
events[60] = 'Successful Marketing Campaign'
events[80] = 'Customer Service Issue'
sales = 100 + 50 * np.random.randn(num_days) #Simulate Sales impacted by sentiment
df = pd.DataFrame({'Date': dates, 'Sentiment': sentiment_scores, 'Event': events, 'Sales': sales})
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for synthetic data, but this section is crucial for real-world datasets.
# --- 3. Analysis ---
# Calculate rolling average sentiment to smooth out daily fluctuations
df['Rolling_Sentiment'] = df['Sentiment'].rolling(window=7).mean()
# Identify days with significant sentiment shifts (e.g., > 0.2 change from previous day)
df['Sentiment_Change'] = df['Rolling_Sentiment'].diff()
df['Significant_Change'] = np.abs(df['Sentiment_Change']) > 0.2
# --- 4. Visualization ---
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Rolling_Sentiment'], label='Rolling Average Sentiment')
plt.scatter(df[df['Significant_Change']]['Date'], df[df['Significant_Change']]['Rolling_Sentiment'], color='red', label='Significant Sentiment Changes')
for i, txt in enumerate(df[df['Significant_Change']]['Event']):
    if txt != 'No Event':
        plt.annotate(txt, (df[df['Significant_Change']]['Date'].iloc[i], df[df['Significant_Change']]['Rolling_Sentiment'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('Date')
plt.ylabel('Rolling Average Sentiment')
plt.title(f'Brand Sentiment Volatility for {brand_name}')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'brand_sentiment_volatility.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(10,6))
sns.regplot(x='Sentiment', y='Sales', data=df)
plt.xlabel('Sentiment Score')
plt.ylabel('Sales')
plt.title(f'Sales vs. Sentiment for {brand_name}')
plt.grid(True)
plt.tight_layout()
output_filename2 = 'sales_vs_sentiment.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")
#Further analysis could involve event-based analysis, topic modeling on social media text to understand the context of events, etc.  This is a basic example.