# Election Model Using Bayesian Heirarchal Regression

I created an election model using hierarchal regression of the latest polls for 2024 US Presidential Election. The visualizations of this data will be displayed on my website [here](https://alexbass.me).

![](data_pipeline.png)

I visualized my processing pipeline above. This runs daily via github actions.

1. Load the polling data freely available on fivethirtyeight from a URL.
2. Process the data in Python
3. Estimate the model in Python (pymc)
4. Save the daily outputs in this repository
5. Later in the day, my website loads these outputs and generates three figures displayed on my website.

I wrote up a description of my modeling methodology here.

Feel free to clone my work or reach out if there are questions.
