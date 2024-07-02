# Designing MLOps pipelines

---

!!! danger
    Module is still under development

"Machine learning engineering is 10% machine learning and 90% engineering." - *Chip Huyen*

We highly recommend that you read the book
*Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications* by Chip Huyen which gives
an fantastic overview of the thought processes that goes into designing moder machine learning systems.

# The stack

Have you ever encountered the concept of **full stack developer**. A full stack developer is an developer who can
both develop client and server software or in more general terms, it is a developer who can take care of the complete
developer pipeline.

Below is seen an image of the massive amounts of tools that exist within the MLOps umbrella.
<figure markdown>
![Overview](../figures/tool_landscape.PNG){ width="1000" }
<figcaption>
<a href=" https://mlops.neptune.ai/"> Image credit </a>
</figcaption>
</figure>

## Visualizing the design

<figure markdown>
![Overview](../figures/diagrams/mlops_canvas.drawio){ width="1000" }
<figcaption>
</figcaption>
</figure>

In this course we have chosen to make our own canvas, but it is highly inspired by
[Machine Learning Canvas](https://madewithml.com/static/templates/ml-canvas.pdf) from the
[Made with ML course](https://madewithml.com/) and the
[Machine Learning Canvas](https://www.ownml.co/machine-learning-canvas) by Louis Dorard.

<!-- markdownlint-disable -->

!!! note "Editing the canvas"

    The canvas is made in [draw.io](https://app.diagrams.net/), which I use for all my diagrams. You can easily begin
    editing the canvas by clicking
    [here](https://app.diagrams.net/?client=1#Uhttps%3A%2F%2Fraw.githubusercontent.com%2FSkafteNicki%2Fdtu_mlops%2Fdesign%2Ffigures%2Fdiagrams%2Fmlops_canvas.drawio#%7B%22pageId%22%3A%22pnmmrmMmQEoPrk8f0Ppy%22%7D) which will open the web version of draw.io with the canvas loaded.

<!-- markdownlint-restore -->

??? example "Retail Company - Inventory Management Optimization"

    `Background`
    : The retail company, XYZ Retail, faces challenges with overstocking and stockouts, leading to lost sales and
    increased holding costs. They aim to use machine learning to optimize inventory levels across their network of
    stores.

    `Value Proposition`
    : The ML solution will predict demand more accurately, reducing excess inventory and stockouts. This will enhance
    customer satisfaction by ensuring product availability and improve the company's profitability by lowering holding
    costs.

    `Objectives`
    : The primary goal is to reduce inventory costs by 15% while maintaining product availability at 98%. Secondary
    objectives include improving demand forecast accuracy by 20% and reducing lost sales due to stockouts by 10%.

    `Data Collection`
    : Data will be collected from sales transactions, inventory logs, and supply chain records. Additional data sources
    include market trends, seasonal factors, and promotional activities. Data will be gathered daily to ensure timely
    updates to the model.

    `Data Governance`
    : XYZ Retail will implement strict data governance policies to ensure data quality, accuracy, and security. Data
    privacy measures will comply with relevant regulations, and access controls will be enforced to protect sensitive
    information.

    `Modelling`
    : The model will utilize time-series forecasting techniques, such as ARIMA and LSTM networks, to predict future
    demand. Feature engineering will include variables like past sales, promotional events, and external factors such
    as holidays and weather conditions.

    `Metrics & Evaluation`
    : Model performance will be evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error
    (RMSE). Additionally, inventory turnover rates and the frequency of stockouts will be monitored to assess the
    model's real-world impact.

    `Model Governance`
    : A dedicated team will monitor the model's performance and update it as necessary. Regular retraining will be
    scheduled to incorporate new data, and any significant deviations in performance will trigger a review and potential
    model adjustments.

    `Inference`
    : The model will be deployed in a cloud environment, allowing for real-time predictions. It will integrate with the
    company's inventory management system to provide actionable insights, such as reorder points and quantities for each
    product.

    `Decision`
    : The model's predictions will inform purchasing and inventory decisions, with automated alerts for reordering
    products. Inventory managers will have the final say in decisions, ensuring a balance between model recommendations
    and practical considerations.

    `Feedback`
    : Feedback will be collected from inventory managers and sales data to assess the model's accuracy and relevance.
    This feedback loop will help identify areas for improvement and ensure the model adapts to changing business
    conditions.

    `Lifetime`
    : The model's lifecycle will include phases for development, testing, deployment, and maintenance. It will be
    versioned to track changes and improvements, with plans for decommissioning and replacement as the business evolves
    and new technologies emerge.

??? example "Healthcare Company - Predicting Patient Readmissions"

    `Background`
    : ABC Healthcare faces challenges with high patient readmission rates, which lead to increased costs and reduced
    patient satisfaction. They aim to use machine learning to identify patients at high risk of readmission within 30
    days of discharge.

    `Value Proposition`
    : The ML solution will predict readmission risk, enabling proactive interventions. This will improve patient
    outcomes by providing timely care and reduce costs associated with readmissions, enhancing overall healthcare
    efficiency.

    `Objectives`
    : The primary goal is to reduce the 30-day readmission rate by 20%. Secondary objectives include improving patient
    care coordination and achieving cost savings of $1 million annually by reducing unnecessary readmissions.

    `Data Collection`
    : Data will be collected from electronic health records (EHRs), including patient demographics, medical history,
    treatment plans, and post-discharge follow-up information. Additional data sources include social determinants of
    health and patient feedback surveys.

    `Data Governance`
    : ABC Healthcare will implement robust data governance policies to ensure the quality, privacy, and security of
    patient data. Compliance with HIPAA and other relevant regulations will be strictly maintained, with controlled
    access to sensitive data.

    `Modelling`
    : The model will use logistic regression and random forests to predict readmission risk based on patient health
    data. Feature engineering will include factors like comorbidities, prior hospitalizations, discharge instructions,
    and follow-up adherence.

    `Metrics & Evaluation`
    : Model performance will be evaluated using metrics such as Area Under the ROC Curve (AUC-ROC), precision, recall,
    and F1 score. Additionally, the actual reduction in readmission rates and cost savings will be monitored to measure
    real-world impact.

    `Model Governance`
    : A multidisciplinary team will oversee model monitoring and maintenance. Regular updates and retraining will be
    performed to incorporate new patient data and medical insights, ensuring the model remains accurate and relevant.

    `Inference`
    : The model will be deployed within the hospital’s IT infrastructure, providing real-time risk scores for patients
    at discharge. Integration with the system will allow healthcare providers to access these predictions during
    routine care.

    `Decision`
    : The model’s predictions will inform discharge planning and post-discharge follow-up strategies. Healthcare
    providers will use these insights to tailor interventions, such as arranging home visits, coordinating with primary
    care, and providing patient education.

    `Feedback`
    : Feedback will be gathered from healthcare providers and patient outcomes to assess the model’s accuracy and
    usefulness. This continuous feedback loop will help refine the model and improve its predictive capabilities.

    `Lifetime`
    : The model’s lifecycle will encompass development, testing, deployment, and continuous improvement. Regular
    reviews will be conducted to ensure the model adapts to new medical knowledge and patient care practices, with
    plans for eventual decommissioning and replacement.

??? example "Energy Company - Optimizing EV Charging Network"

    `Background`
    : GreenCharge Energy is expanding its network of EV charging stations. They aim to use machine learning to determine
    optimal locations for new charging stations and predict usage patterns to ensure efficient operation.

    `Value Proposition`
    : The ML solution will identify high-demand areas and optimize the placement of charging stations, leading to
    increased usage and customer satisfaction. It will also help in resource allocation, reducing downtime and
    operational costs.

    `Objectives`
    : The primary goal is to increase the utilization rate of charging stations by 25%. Secondary objectives include
    reducing downtime by 15% and ensuring that 90% of customers have access to a charging station within a 10-minute
    drive.

    `Data Collection`
    : Data will be collected from existing charging stations, including usage patterns, location data, and maintenance
    logs. Additional data sources include traffic patterns, demographic information, and EV ownership statistics in
    different areas.

    `Data Governance`
    : GreenCharge Energy will implement comprehensive data governance policies to ensure the quality, security, and
    privacy of collected data. Compliance with relevant regulations, such as GDPR, will be strictly enforced.

    `Modelling`
    : The model will use clustering algorithms and regression analysis to identify optimal locations and predict usage
    patterns. Feature engineering will include factors such as proximity to highways, commercial areas, and residential
    neighborhoods.

    `Metrics & Evaluation`
    : Model performance will be evaluated using metrics such as Mean Absolute Error (MAE) for usage predictions and the
    accuracy of location recommendations. Utilization rates and customer satisfaction surveys will also be monitored.

    `Model Governance`
    : A dedicated team will monitor the model’s performance and update it as needed. Regular retraining will be
    conducted to incorporate new data, and any significant deviations in performance will trigger a model review and
    adjustment.

    `Inference`
    : The model will be deployed in a cloud-based environment, providing real-time predictions for usage and optimal
    placement. Integration with the company's planning and operational systems will enable dynamic decision-making.

    `Decision`
    : The model’s insights will guide the placement of new charging stations and the maintenance schedule for existing
    ones. Operational decisions, such as the allocation of resources for peak times, will be informed by the model's
    predictions.

    `Feedback`
    : Feedback will be collected from station usage data, customer feedback, and maintenance logs to continuously refine
    the model. This feedback loop will help improve the accuracy of predictions and the efficiency of operations.

    `Lifetime`
    : The model’s lifecycle will include phases for development, testing, deployment, and maintenance. Version control
    will be implemented to track changes, with plans for eventual decommissioning and replacement as new technologies
    and data become available.
