# **Architectural Design and Implementation of Quantitative Sports Wagering Systems: Multi-Season Data Acquisition and Model Integration within AI-Native Environments**

The intersection of quantitative finance and sports analytics has undergone a transformative shift with the advent of high-fidelity data streams and AI-augmented development environments. For the professional developer seeking to construct a robust betting model for the National Basketball Association (NBA) and National Collegiate Athletic Association Basketball (NCAAB), the primary challenge lies not merely in the application of machine learning algorithms but in the sophisticated engineering of the data pipeline. Successful predictive systems for the 2023-2024 through 2025-2026 seasons require a multi-faceted approach to data acquisition that balances cost, latency, and historical depth. When implementing these systems within an environment such as Cursor, the architect must leverage integrated development features—ranging from .cursorrules context management to automated agentic coding—to ensure that the model can ingest, store, and act upon diverse datasets including closing odds, real-time results, and granular team performance metrics.

## **Foundational Data Acquisition Strategies for Professional Wagering**

The initial phase of model development necessitates the selection of data sources that provide coverage for the previous two seasons while offering a pathway for real-time integration during active competition. The ecosystem for basketball data is stratified into programmatic API services, community-driven repositories, and specialized scraping frameworks, each serving a distinct role in the training and execution of predictive models.

### **Programmatic Access and API Architectures**

The most reliable method for obtaining structured, real-time betting information is through professional-grade APIs that aggregate lines from a multitude of global bookmakers. For developers operating under budget constraints, The Odds API provides a tiered access model that is particularly well-suited for the prototyping phase of a betting model. The free usage plan offers 500 credits per month, covering over 70 sports and including major markets for the NBA and NCAAB.1 This service allows for the retrieval of live and upcoming game start times, home and away team designations, and odds for moneyline (h2h), spreads, and totals.1 For historical context, the platform maintains snapshots starting from late 2020 for featured markets, providing the two-year lookback required for backtesting models against the 2023-2024 and 2024-2025 seasons.1

Beyond simple betting lines, deeper statistical insights are required to fuel the feature engineering process. The balldontlie NBA API serves as a comprehensive resource for historical and current player and team statistics, with records extending back to 1979\.5 This RESTful service is essential for calculating performance trends that are independent of market price, such as true shooting percentages or defensive rating shifts over a rolling window. For a more comprehensive solution that merges stats and odds, Sports Game Odds offers an API that includes box scores, period-level scoring, and key player statistics for all NBA games, though its historical depth is primarily accessible through paid tiers or trial extensions.6

### **Community Repositories and Static Dataset Landscapes**

When the model requires massive volumes of historical data for initial training—where the overhead of thousands of API calls might be prohibitive—static datasets on platforms like Kaggle become indispensable. The current basketball data landscape features several high-usability datasets specifically designed for machine learning enthusiasts. For example, a meticulously maintained NBA dataset updated daily provides player box scores for every game since 1949, franchise histories, and up-to-date schedule information for the 2024-2025 season.7 These datasets are typically distributed in .csv or .sql formats, facilitating immediate ingestion into the Cursor-based project structure.7

| Dataset Provider | Primary League Coverage | Data Category | Historical Depth | Format |
| :---- | :---- | :---- | :---- | :---- |
| **The Odds API** | NBA, NCAAB, Euroleague | Odds & Basic Scores | From late 2020 | JSON 1 |
| **balldontlie** | NBA | Team & Player Stats | From 1979 | JSON 5 |
| **Eoin Moore (Kaggle)** | NBA | Box Scores & Bios | From 1949 | CSV/SQL 7 |
| **Robby Peery (Kaggle)** | NCAAB | Play-by-Play & Odds | 2023-2024 Season | CSV 9 |
| **Sports Game Odds** | NBA | Live Odds & Results | Current | JSON 6 |

For collegiate basketball, the data environment is more fragmented due to the sheer volume of Division 1 teams. However, specific repositories target the 2023-2024 and 2024-2025 seasons with high precision. Datasets including pre-game spread and total lines for approximately 90% of NCAAB games are available, often paired with conditional win probabilities sourced from major broadcasters.9 This granularity allows for the development of models that account for the "March Madness" effect, where historical performance often deviates from tournament output.10

### **Automated Scraping Frameworks and Logic**

When static datasets are unavailable or official APIs lack the required depth for a specific market, browser automation serves as the primary mechanism for data harvesting. OddsPortal is a critical target for this activity, as it hosts one of the largest archives of historical betting odds. However, because the site uses dynamic JavaScript rendering, standard HTTP requests are insufficient.11 Developers must utilize libraries such as Selenium or Playwright within their Python environment to simulate a real user's browser, allowing the script to render the page and extract the data from the DOM.11

The OddsHarvester application exemplifies this approach, providing a command-line interface to scrape upcoming and historical matches from OddsPortal across multiple sports, including the NBA and NCAAB.12 These tools can be configured to retrieve closing odds—which are generally preferred for model evaluation—or opening odds, which are essential for identifying market movement and the direction of professional "sharp" money.13 Managing these scrapers within Cursor involves creating a dedicated src/Process-Data directory where scripts for fetching, cleaning, and merging data are version-controlled and executed.15

## **Database Architecture and Information Engineering in Cursor**

Storing two seasons of NBA and NCAAB data for real-time decision-making requires a relational database schema that prioritizes query performance and historical data integrity. SQLite is the optimal engine for this use case because of its serverless, single-file architecture, which integrates seamlessly with the Cursor project folder and allows the AI agent to interact with the data directly through SQL magic commands.15

### **Relational Schema Design for Quantitative Models**

A sophisticated betting database must be normalized to prevent data redundancy while allowing for complex joins between statistical performance and betting market pricing. The core architecture should center around a Games table that acts as the primary join point for all other information clusters.17

#### **The Core Entitlement Tables**

The database should begin with static tables defining the sports and the bookmakers being monitored. The Sports table stores unique identifiers such as basketball\_nba and basketball\_ncaab, while the Casinos or Bookmakers table maps API-provided keys to human-readable names like "FanDuel" or "DraftKings".15

#### **The Games and Results Schema**

The central Games table records the architectural facts of each matchup:

* game\_id: A unique primary key, often derived from the source API's unique identifier.  
* sport\_id: A foreign key linking to the league type.  
* home\_team, away\_team: String identifiers for the participants.  
* game\_time: A DATETIME field, standardized to UTC to handle cross-timezone schedules.17

Linked to this is the Scores table, which may store partial results (halftime, period-level) or final outcomes. For NBA models, tracking the final score alongside the update time is crucial for validating that the betting decision was made based on pre-match information, thus avoiding data leakage.17

#### **The Multi-Market Odds Schema**

The most complex component is the Odds table, which must store the temporal movement of lines. A professional schema will capture:

* odds\_id: Primary key.  
* game\_id: Foreign key to the matchup.  
* bookmaker\_id: Foreign key to the source.  
* market\_type: Enum for "moneyline", "spread", or "total".  
* home\_price, away\_price: The decimal or American odds.  
* line\_value: The numerical spread (e.g., \-5.5) or over/under total (e.g., 220.5).  
* timestamp: The exact moment the line was captured, allowing the model to analyze line movement over time.4

### **Integrating Data Storage with Cursor AI Agents**

Cursor's ability to "index" a codebase means that the AI agent can be made aware of the SQLite database schema and the structure of local CSV files. By placing these files in a predictable directory, such as /data/historical/, and providing a .cursorrules file that describes the schema, the developer enables the AI to write high-precision data analysis scripts without manual intervention.19

The AI agent can utilize MCP (Model Context Protocol) servers or built-in extensions to connect directly to the database, suggesting optimizations for window functions or complex joins that calculate rolling averages for team performance.16 For example, the agent can be prompted to "calculate the against-the-spread (ATS) cover rate for all NBA home favorites in the 2024-2025 season," and it will generate the necessary SQL and Python code to extract that insight from the stored data.16

## **Implementation Framework in the Cursor IDE Environment**

Building a betting model in Cursor is not merely about writing code; it is about establishing a workflow where the AI acts as a creative partner and a meticulous data engineer. The environment supports several levels of rule-based guidance that shape the AI's behavior to match the project's specific architectural needs.

### **Leveraging Multi-Level Rules for Sports Data Science**

To ensure the AI maintains the integrity of the betting model, developers should implement a hierarchy of rules. At the repository level, a .cursor/rules/index.mdc file (configured to "Always" apply) should provide a project overview, the technology stack (e.g., Python 3.11, XGBoost, SQLite), and specific architectural patterns like functional programming for data cleaning.20

| Rule Level | Implementation Method | Purpose in Betting Model |
| :---- | :---- | :---- |
| **Global Rules** | Cursor Settings \> Rules for AI | Base preferences (e.g., "Always use functional programming," "Comments in English").20 |
| **Project Rules** | .cursor/rules/index.mdc | Defining the SQLite schema and library choices (e.g., "Use pandas for data manipulation").20 |
| **Contextual Rules** | .cursor/rules/\*.mdc | Specific guidance for tasks like "Backtesting" or "Scraping" that only activate when relevant.20 |
| **Agent Instructions** | AGENTS.md | High-level behavioral guidance for the AI agent during long-running tasks.19 |

For a betting model, a contextual rule might specify that "When working with odds data, always calculate the implied probability before performing any comparison" or "Ensure all timestamps are converted to the user's local timezone before display".20 These rules prevent the AI from making common domain-specific errors, such as comparing American odds directly without accounting for the vig or the different directions of positive and negative values.

### **Agentic Workflows for Model Development and Iteration**

The Cursor AI agent can automate the "plumbing" of a sports betting app—fetching data, running tests, and iterating on model performance. A standard workflow for adding a new feature, such as a "player injury tracker," involves prompting the agent to research the existing codebase, identify relevant files, and propose an implementation plan before writing any code.22 This deliberative approach is critical in betting models where small logic errors in data processing can lead to significant financial discrepancies.

The agent is also capable of performing Test-Driven Development (TDD). A developer might ask the agent to "write tests for a Kelly Criterion staking function where the bankroll fraction is limited to 1% per unit," then instruct it to run the tests and iterate on the implementation until they pass.22 This process ensures that the core mathematical logic of the betting engine is sound and reproducible.

## **Advanced Feature Engineering for Basketball Analytics**

The transition from raw data to a predictive signal requires the creation of sophisticated features that capture the nuances of professional and collegiate basketball. For the 2023-2026 seasons, several advanced metrics have emerged as the primary drivers of model accuracy.

### **Efficiency and Possession-Based Metrics**

In both the NBA and NCAAB, traditional counting statistics are often skewed by the tempo of the game. Professional models prioritize "Adjusted Efficiency" metrics, which quantify a team's performance per 100 possessions while adjusting for the strength of the opponent.10

* **Adjusted Offensive Efficiency (AdjO):** The points a team would score against an average defense.  
* **Adjusted Defensive Efficiency (AdjD):** The points a team would allow against an average offense.  
* **Efficiency Ratio:** A derived feature calculated as ![][image1]. Historical analysis of the 2025 March Madness field showed that teams with an efficiency ratio significantly higher than their tournament seeding (e.g., Duke as a \#1 seed in KenPom but \#3 in AP) were the most likely to cover deep tournament runs.10  
* **Tempo and Possession:** The number of plays a team executes per game. Possession is calculated as the total number of plays, while pace is the frequency of those opportunities.24 These are the primary inputs for over/under models, as they define the total opportunity for scoring in a matchup.10

### **Situational and Environmental Features**

Machine learning models excel at identifying patterns that human analysts might overlook. Advanced features should include the "Days Rest" variable, which accounts for fatigue, particularly in the NBA's "back-to-back" game scenarios.15 Environmental factors like "Rainy Days" (for games in certain arenas) or "Long Road Trips" are also quantifiable through data processing pipelines in Cursor, allowing the model to learn how specific conditions impact team performance.26

For collegiate basketball, a custom metric known as "Upset Propensity" can be engineered. This feature identifies teams from smaller conferences that possess elite efficiency metrics (e.g., high Effective Field Goal Percentage) when playing against power-conference teams with high turnover rates.23 In the 2025 tournament, this specific feature allowed models to pick Colorado State's upset over Memphis correctly.23

### **Market-Derived Features and Line Movement**

The betting market itself acts as a massive information aggregator. Features derived from market behavior are often the most predictive.

* **Closing Line Value (CLV):** The difference between the price at which a bet was placed and the final closing price.  
* **Line Movement Delta:** The shift from the opening line to the closing line. Analyzing these variations across multiple bookmakers helps identify "sharp" money influence.13  
* **Vig-Free Implied Probability:** Removing the bookmaker's "hold" to find the "fair" probability of an outcome is a standard preprocessing step in models using libraries like sportsbooklib.27

## **Predictive Modeling Architectures and Training Pipelines**

With the data engineering and Cursor environment established, the focus shifts to the machine learning architecture. The previous two seasons of data provide a rich foundation for training ensemble models and neural networks.

### **XGBoost and Ensemble Methods for Classification**

The kyleskom NBA betting project serves as a premier example of a high-performing model architecture. It utilizes XGBoost and Neural Network models to predict game winners and point totals.15 The pipeline follows a structured sequence:

1. **Data Collection:** Pulling daily stats and odds into separate SQLite databases.15  
2. **Feature Merging:** A script titled Create\_Games merges stats, odds, scores, and days-rest into a training dataset.15  
3. **Hyperparameter Tuning:** Using scripts like XGBoost\_Model\_ML to run multiple trials and optimize the model for specific calibration methods like sigmoid.15  
4. **Calibration:** Ensuring that the model's raw probability outputs match real-world outcomes. A well-calibrated model will have a black line that straddles the ![][image2] diagonal on a calibration plot, indicating that its "60% sure" predictions win 60% of the time.18

### **Neural Networks and Time-Series Analysis**

While XGBoost is dominant for tabular data, Artificial Neural Networks (ANN) and Long Short-Term Memory (LSTM) networks are occasionally used to capture the sequential nature of player performance.15 In the context of March Madness 2025, logistic regression models were found to be surprisingly effective despite their simplicity, often outperforming complex LSTMs when the dataset was limited to efficiency metrics.23 This highlights the importance of matching the model complexity to the signal-to-noise ratio of the data.

| Model Type | Evaluation Metric | Threshold for Professional Use | Purpose |
| :---- | :---- | :---- | :---- |
| **XGBoost** | Accuracy | \> 60% (Raw Accuracy) 18 | Moneyline/Winner Prediction |
| **Neural Network** | Log Loss | \< 0.30 (Caution is valued) 23 | Totals and Spread Coverage |
| **Logistic Regression** | ROC-AUC | \> 0.90 (Satisfying Rank power) 23 | Probabilistic Win Estimation |
| **Ensemble** | RPS | Lower is better (multi-class) 28 | Tournament Bracket Prediction |

### **Cross-Validation and Data Leakage Prevention**

One of the most critical aspects of model training is preventing "data leakage"—the accidental inclusion of future information in the training of past events. For sports models, this is addressed by generating "lagged variables," where the inputs for a game are based only on the statistics available at the time the bet would be placed.18 The dataset is typically split into a training set (80%) and a test set (20%) to ensure the model generalizes to unseen data.18 Advanced cross-validation techniques, such as 20-fold cross-validation, are applied to the historical seasons to confirm that the model's predictive accuracy is statistically significant.29

## **Risk Management and Staking Frameworks**

A predictive model is only as valuable as the staking strategy it employs. Professional-grade betting systems integrate the model's probability outputs with bankroll management algorithms to ensure long-term profitability.

### **The Kelly Criterion and Unit Sizing**

The Kelly Criterion is the gold standard for sizing wagers based on the perceived edge. It calculates the fraction of the bankroll to bet by considering the decimal odds and the model's estimated win probability.15 For the 2024-2025 NBA season, model developers often implemented a "Fractional Kelly" approach—betting only a portion (e.g., 25% or 50%) of the calculated Kelly amount—to account for model uncertainty and reduce bankroll volatility.15

### **Expected Value (EV) and Market Exploitation**

The primary goal of a model is not to pick winners but to find "Positive EV" bets—situations where the model's probability of an outcome is higher than the probability implied by the bookmaker's odds.18 If a bookmaker prices an NBA team at \+100 (50% implied), but the model estimates a 63% win probability, that is a significant probabilistic signal.18 Real-time prediction scripts like main.py in Cursor-based projects fetch today's schedule, build matchup features, load the trained models, and print the expected value for each game alongside the recommended Kelly stake sizing.15

| Betting Metric | Definition | Importance |
| :---- | :---- | :---- |
| **Implied Probability** | **![][image3]** | The market's baseline expectation. |
| **Model Probability** | Sigmoid output of the ML model | The independent forecast. |
| **Expected Value (EV)** | **![][image4]** | The measure of a "good" bet.30 |
| **Kelly Criterion** | Optimal stake for growth | Science-based bankroll management.30 |
| **Hold (Vig)** | The bookmaker's commission | Must be removed to find fair price.27 |

## **Comparative Analysis of the 2023-2026 Basketball Seasons**

The past two seasons have provided a distinct statistical signature that models must account for when making decisions for the 2025-2026 campaign.

### **NBA Trends: Win Totals and ATS Results**

The 2024-2025 NBA season was characterized by the dominance of specific teams against the spread (ATS), with the Boston Celtics and Charlotte Hornets yielding significant returns on investment for data-driven bettors.31 Conversely, teams like the Chicago Bulls and Sacramento Kings consistently underperformed relative to market expectations.31

#### **2025-2026 Preseason Projections and Early Outcomes**

Preseason win totals for the 2025-2026 season highlight the market's perception of league hierarchy. The Oklahoma City Thunder opened with a league-high win total of 62.5, while the Brooklyn Nets were projected for a league-low 19.5 wins.32

| Team | Opening Win Total (25-26) | Spread ROI (Early 25-26) | Win % (Preseason Odds) |
| :---- | :---- | :---- | :---- |
| **Oklahoma City Thunder** | 62.5 | \-4.55% | 49-15 (Under-pacing) 31 |
| **Boston Celtics** | 41.5 | \+12.67% | 41-21 (Over-pacing) 31 |
| **Charlotte Hornets** | 27.5 | \+23.70% | 32-31 (Over-pacing) 31 |
| **Brooklyn Nets** | 19.5 | \-15.66% | 15-47 (Under-pacing) 31 |
| **Phoenix Suns** | 30.5 | \+16.21% | 35-27 (Over-pacing) 31 |

The early data for the 2025-2026 season indicates that teams projected with low win totals, like the Hornets and Suns, have significantly outperformed their ATS expectations, providing fertile ground for models that prioritize "value" over raw team strength.31

### **NCAAB Trends: Tournament Dynamics and Efficiency**

Collegiate basketball in the 2024-2025 season saw the rise of Florida as the National Champion, defeating the highly-touted Duke squad in the tournament.35 This outcome was reflected in the KenPom rankings, where Duke, Houston, Florida, and Auburn consistently occupied the top four spots across adjusted offensive and defensive efficiency metrics.10

#### **Closing Spread Records and Coverage**

Analysis of NCAAB closing lines from 2003 through the 2025-2026 season shows a high level of market efficiency. For example, teams that are heavy underdogs (30-plus point spreads) in the Round of 64 historically cover only 39.1% of the time, suggesting that bookmakers tend to slightly undervalue the "blowout" potential of top-seeded teams in the tournament.36

| NCAAB Metric | 2024-2025 Result / Leader | Implications for 2025-2026 Models |
| :---- | :---- | :---- |
| **National Champion** | Florida 35 | Efficiency metrics were superior to media polls.10 |
| **Wooden Award** | Cooper Flagg 35 | Individual player usage rates remained predictive. |
| **Top KenPom Defense** | Duke 10 | High defensive efficiency correlated with deep runs. |
| **Upset Rate (5-7 Seeds)** | \~4 Major Upsets 37 | Seed-based models must account for "Cinderella" risk. |

The data for the 2025-2026 season shows that conference champion races are tightening, with Duke and Arizona leading their respective conferences in preseason odds, but underperforming in early ATS metrics as the market adjusts to high-usage freshman talent.38

## **Ethical Considerations and Operational Constraints**

The development of sophisticated betting systems is bound by both the technical constraints of the data providers and the legal requirements of the jurisdictions where the bets are placed.

### **Data Usage and API Terms**

API providers like The Odds API and API-SPORTS explicitly state that their services are for personal, analytical, or research purposes.40 Developers must be careful not to "repackage or redistribute" the data as a standalone product, which could lead to the revocation of API keys.40 Furthermore, when using free tiers, the quota of 500 credits per month requires strategic scheduling of API calls—perhaps limiting requests to just before game starts and once daily for historical updates—to avoid premature quota depletion.2

### **Managing Overfitting and Market Efficiency**

A persistent risk in sports modeling is "overfitting," where the model identifies false relationships in historical data that do not repeat in the future.26 This risk is heightened in basketball because of its high variance and the "noise" created by injuries or lineup changes. Professional models address this through "Sound Validation Methods," ensuring that the model remains relevant over time by continuously learning from new data.26

Furthermore, as AI-powered prediction tools become more common, the market becomes more efficient. Casual bettors typically have a \~50% pick rate, but the use of ML insights can push this toward 60%—the threshold where the bettor begins to overcome the bookmaker's juice (usually \-110 or 1.91 decimal).42 However, as more participants use similar data sources and modeling techniques, the edges become smaller, requiring more specialized features—like player tracking data or sentiment analysis—to maintain a profitable margin.42

## **Summary of Implementation Best Practices for Cursor**

The successful integration of these diverse data streams into a Cursor-based project follows a clear hierarchy of implementation:

1. **Establish the Data Infrastructure:** Create a Python environment that utilizes nba\_api and CBBpy for stats, and The Odds API or custom scrapers for betting lines.2  
2. **Design the Persistence Layer:** Implement a normalized SQLite database within the project folder to store two seasons of game results and odds movement.15  
3. **Optimize for AI-Native Development:** Use .cursorrules and .mdc files to define the database schema and architectural standards, enabling the AI agent to act as a high-precision developer.19  
4. **Engineer Predictive Features:** Prioritize Adjusted Efficiency metrics, possession rates, and market-derived features like line movement.10  
5. **Implement Staking and Risk Management:** Use the Kelly Criterion and Expected Value analysis to ensure that the model's predictions are translated into disciplined wagering decisions.30

By following this architectural framework, the quantitative analyst can construct a betting system that is not only statistically sound but also operationally efficient within the modern development landscape of the 2024-2026 seasons.

#### **Works cited**

1. NCAA Basketball Odds API, accessed March 6, 2026, [https://the-odds-api.com/sports-odds-data/ncaa-basketball-odds.html](https://the-odds-api.com/sports-odds-data/ncaa-basketball-odds.html)  
2. The Odds API: Sports Odds API, accessed March 6, 2026, [https://the-odds-api.com/](https://the-odds-api.com/)  
3. NCAA Basketball Odds API, accessed March 6, 2026, [https://the-odds-api.com/sports/ncaab-odds.html](https://the-odds-api.com/sports/ncaab-odds.html)  
4. Historical Sports Odds Data API, accessed March 6, 2026, [https://the-odds-api.com/historical-odds-data/](https://the-odds-api.com/historical-odds-data/)  
5. 20 Best Free Sports Datasets for ML 2025 \- Unidata, accessed March 6, 2026, [https://unidata.pro/blog/best-free-sports-datasets-ml/](https://unidata.pro/blog/best-free-sports-datasets-ml/)  
6. NBA Odds API | Free Odds API \- Sports Game Odds, accessed March 6, 2026, [https://sportsgameodds.com/nba-odds-api/](https://sportsgameodds.com/nba-odds-api/)  
7. NBA Historical Dataset: Box Scores, Player Stats, and Game Data (1949–Present) \- Reddit, accessed March 6, 2026, [https://www.reddit.com/r/algobetting/comments/1i2gdc9/nba\_historical\_dataset\_box\_scores\_player\_stats/](https://www.reddit.com/r/algobetting/comments/1i2gdc9/nba_historical_dataset_box_scores_player_stats/)  
8. Complete NBA Box Score dataset, updated daily : r/fantasybball \- Reddit, accessed March 6, 2026, [https://www.reddit.com/r/fantasybball/comments/1ocrud1/complete\_nba\_box\_score\_dataset\_updated\_daily/](https://www.reddit.com/r/fantasybball/comments/1ocrud1/complete_nba_box_score_dataset_updated_daily/)  
9. College Basketball PbP 23-24 \- Kaggle, accessed March 6, 2026, [https://www.kaggle.com/datasets/robbypeery/college-basketball-pbp-23-24](https://www.kaggle.com/datasets/robbypeery/college-basketball-pbp-23-24)  
10. ai-tests/March Madness/Gemini at main \- GitHub, accessed March 6, 2026, [https://github.com/jaldps/ai-tests/blob/main/March%20Madness/Gemini](https://github.com/jaldps/ai-tests/blob/main/March%20Madness/Gemini)  
11. Web Scraping oddsportal.com \- ScrapeHero, accessed March 6, 2026, [https://www.scrapehero.com/scraping-odds-portal/](https://www.scrapehero.com/scraping-odds-portal/)  
12. jordantete/OddsHarvester: A python app designed to scrape and process sports betting data directly from oddsportal.com \- GitHub, accessed March 6, 2026, [https://github.com/jordantete/OddsHarvester](https://github.com/jordantete/OddsHarvester)  
13. Scrape Betting Odds from Oddsportal website \- WebHarvy, accessed March 6, 2026, [https://www.webharvy.com/articles/scraping-oddsportal.html](https://www.webharvy.com/articles/scraping-oddsportal.html)  
14. morrisndurere/Sports-Betting-Data-Scraping \- GitHub, accessed March 6, 2026, [https://github.com/morrisndurere/Sports-Betting-Data-Scraping](https://github.com/morrisndurere/Sports-Betting-Data-Scraping)  
15. kyleskom/NBA-Machine-Learning-Sports-Betting \- GitHub, accessed March 6, 2026, [https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting)  
16. Data Science | Cursor Docs, accessed March 6, 2026, [https://cursor.com/docs/cookbook/data-science](https://cursor.com/docs/cookbook/data-science)  
17. Building a Database for Historical Sports Betting Spreads with the Odds API | by Ben Todd, accessed March 6, 2026, [https://medium.com/@bentodd\_46499/building-a-database-for-historical-sports-betting-spreads-with-the-odds-api-5575fb87d650](https://medium.com/@bentodd_46499/building-a-database-for-historical-sports-betting-spreads-with-the-odds-api-5575fb87d650)  
18. throwawayhub25/Sports-Betting-Model \- GitHub, accessed March 6, 2026, [https://github.com/throwawayhub25/Sports-Betting-Model](https://github.com/throwawayhub25/Sports-Betting-Model)  
19. Rules | Cursor Docs, accessed March 6, 2026, [https://cursor.com/docs/context/rules](https://cursor.com/docs/context/rules)  
20. Cursor IDE Rules for AI: Guidelines for Specialized AI Assistant \- Kirill Markin, accessed March 6, 2026, [https://kirill-markin.com/articles/cursor-ide-rules-for-ai/](https://kirill-markin.com/articles/cursor-ide-rules-for-ai/)  
21. Rules \- Cursor Directory, accessed March 6, 2026, [https://cursor.directory/rules](https://cursor.directory/rules)  
22. Best practices for coding with agents \- Cursor, accessed March 6, 2026, [https://cursor.com/blog/agent-best-practices](https://cursor.com/blog/agent-best-practices)  
23. Machine Learning March Madness \- by Anthony Klemm \- Medium, accessed March 6, 2026, [https://medium.com/@anthony.klemm/machine-learning-march-madness-2b86cb3e21d2](https://medium.com/@anthony.klemm/machine-learning-march-madness-2b86cb3e21d2)  
24. NCAA College Basketball Data: Game Log Spreadsheets with Stats & Odds \- BigDataBall, accessed March 6, 2026, [https://www.bigdataball.com/datasets/ncaa/cbb-data/](https://www.bigdataball.com/datasets/ncaa/cbb-data/)  
25. Building Player Projection Models for Betting \- BettorEdge, accessed March 6, 2026, [https://www.bettoredge.com/post/building-player-projection-models-for-betting](https://www.bettoredge.com/post/building-player-projection-models-for-betting)  
26. How to Use Big Data to Improve Betting Efficiency | Alltegrio, accessed March 6, 2026, [https://alltegrio.com/blog/how-to-use-big-data-to-improve-betting-efficiency/](https://alltegrio.com/blog/how-to-use-big-data-to-improve-betting-efficiency/)  
27. carlzoo/sportsbooklib: Python module for performing sportsbook odds calculations \- GitHub, accessed March 6, 2026, [https://github.com/carlzoo/sportsbooklib](https://github.com/carlzoo/sportsbooklib)  
28. Sminton,+13509 Article+ (PDF) 30287 1 11 20220414 | PDF | Machine Learning \- Scribd, accessed March 6, 2026, [https://www.scribd.com/document/750563240/sminton-13509-Article-PDF-30287-1-11-20220414](https://www.scribd.com/document/750563240/sminton-13509-Article-PDF-30287-1-11-20220414)  
29. Ranking rankings: An empirical comparison of the predictive power of sports ranking methods | Request PDF \- ResearchGate, accessed March 6, 2026, [https://www.researchgate.net/publication/274323737\_Ranking\_rankings\_An\_empirical\_comparison\_of\_the\_predictive\_power\_of\_sports\_ranking\_methods](https://www.researchgate.net/publication/274323737_Ranking_rankings_An_empirical_comparison_of_the_predictive_power_of_sports_ranking_methods)  
30. AI-Driven Sports Betting Insights for 2025: Achieve Higher Accuracy \- Parlay Savant, accessed March 6, 2026, [https://www.parlaysavant.com/insights/leveraging-ai-for-smarter-sports-betting-2025](https://www.parlaysavant.com/insights/leveraging-ai-for-smarter-sports-betting-2025)  
31. NBA Betting Stats \- Against The Spread (ATS) \- EV Analytics, accessed March 6, 2026, [https://evanalytics.com/nba/stats/spread](https://evanalytics.com/nba/stats/spread)  
32. 2025-26 NBA Preseason Odds \- Basketball-Reference.com, accessed March 6, 2026, [https://www.basketball-reference.com/leagues/NBA\_2026\_preseason\_odds.html](https://www.basketball-reference.com/leagues/NBA_2026_preseason_odds.html)  
33. 2025-26 NBA Win Totals Tracker – Find the Best Odds for All 30 Teams, accessed March 6, 2026, [https://www.sportsbettingdime.com/nba/futures/win-totals-best-odds/](https://www.sportsbettingdime.com/nba/futures/win-totals-best-odds/)  
34. NBA Win Totals 2025-26: Over/Under Odds & Predictions \- BetMGM, accessed March 6, 2026, [https://sports.betmgm.com/en/blog/nba/nba-odds-predictions-season-win-totals-bm23/](https://sports.betmgm.com/en/blog/nba/nba-odds-predictions-season-win-totals-bm23/)  
35. Archived College Basketball Futures Odds | Sports Odds History by Covers, accessed March 6, 2026, [https://www.covers.com/sportsoddshistory/college-basketball-odds/](https://www.covers.com/sportsoddshistory/college-basketball-odds/)  
36. 2025 March Madness first-round betting trends, NCAA Tournament odds | FOX Sports, accessed March 6, 2026, [https://www.foxsports.com/stories/college-basketball/2025-march-madness-first-round-betting-trends-ncaa-tournament-odds](https://www.foxsports.com/stories/college-basketball/2025-march-madness-first-round-betting-trends-ncaa-tournament-odds)  
37. \[OC\] March Madness Historical Odds \- Round 1: : r/dataisbeautiful \- Reddit, accessed March 6, 2026, [https://www.reddit.com/r/dataisbeautiful/comments/1bi9srr/oc\_march\_madness\_historical\_odds\_round\_1/](https://www.reddit.com/r/dataisbeautiful/comments/1bi9srr/oc_march_madness_historical_odds_round_1/)  
38. 2025-2026 College Basketball Regular Season Conference Champion Odds | Sports Odds History by Covers, accessed March 6, 2026, [https://www.covers.com/sportsoddshistory/cbb-div/?y=2025-2026\&sa=cbb\&a=reg](https://www.covers.com/sportsoddshistory/cbb-div/?y=2025-2026&sa=cbb&a=reg)  
39. NCAA Basketball Team ATS Trends \- All Games, 2025-2026, accessed March 6, 2026, [https://www.teamrankings.com/ncb/trends/ats\_trends/](https://www.teamrankings.com/ncb/trends/ats_trends/)  
40. Terms and Conditions \- The Odds API, accessed March 6, 2026, [https://the-odds-api.com/terms-and-conditions.html](https://the-odds-api.com/terms-and-conditions.html)  
41. 12 Best Free Sports API Options for Developers in 2025 \- SportsJobs Online, accessed March 6, 2026, [https://www.sportsjobs.online/blogposts/43](https://www.sportsjobs.online/blogposts/43)  
42. Machine Learning Sports Predictions Behind Big Wins \- WSC Sports, accessed March 6, 2026, [https://wsc-sports.com/blog/industry-insights/machine-learning-sports-predictions-behind-big-wins/](https://wsc-sports.com/blog/industry-insights/machine-learning-sports-predictions-behind-big-wins/)  
43. I Built Cursor for Sports Betting : r/NFLstatheads \- Reddit, accessed March 6, 2026, [https://www.reddit.com/r/NFLstatheads/comments/1ojwlk8/i\_built\_cursor\_for\_sports\_betting/](https://www.reddit.com/r/NFLstatheads/comments/1ojwlk8/i_built_cursor_for_sports_betting/)  
44. swar/nba\_api: An API Client package to access the APIs for ... \- GitHub, accessed March 6, 2026, [https://github.com/swar/nba\_api](https://github.com/swar/nba_api)  
45. CBBpy · PyPI, accessed March 6, 2026, [https://pypi.org/project/CBBpy/](https://pypi.org/project/CBBpy/)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAYCAYAAABqWKS5AAABqUlEQVR4Xu2VyytFURSHl7wzkMSAJI+SVzGQGQOJiQEmDAxQwlAGBjIUBq4okZGSUPIoKeWZMjMxUQaYkMgfQB6/ddc+13K6N/dOzg37q6/2Xvvs09r7nLU3kcVisVj+C4Ww3B0EKTDGHfSIClgHE00/A1Z/DQuDcBJewV4Vz4GPsEnFvCAersFFOA0v4AhcgBskufrhBFdN+161mR74AQtUzE0/PA7hkfEQHsB96ONJPzAO20w7gSSHE5gOX0ne5acDVhr5oS5nAKzAO9X3Cr3AUpK8ukl+3wFYpMb9jMEXmKZiDyQLiCb8ZTn5PPeA5hLuqX4JyaQ+FQtGPqyPwCqZFjbr8NYd1HA1c6JcFA7OiotVLBiNcDgCO2VaSOLgKGyBsfAZLqnxVjMWgCc8wQnT5+PxnKSAvaaWZNN48/iU4zYXMJNKctpwvt9ohzdwHm7Bd7isH/CITLgNp+AMbCbJaw5uwrLAkwY+jpJJVpQLa0hWzJ8oWugCTYLZqh+AE76Guyp2Cs8oejdr2GTBN9hAsvuzJDcanyK/Ar5Jd0j+8SGSgrVY/iKfAb1VIbQHVF0AAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,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>