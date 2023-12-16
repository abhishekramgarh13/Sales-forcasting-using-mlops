import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main


def main():
    st.title("End to End Sales Forcasting Pipeline with ZenML")

    # high_level_image = Image.open("_assets/high_level_overview.png")
    # st.image(high_level_image, caption="High Level Pipeline")

    # whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")

    # st.markdown(
    #     """ 
    # #### Problem Statement 
    #  The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    # )
    # st.image(whole_pipeline_image, caption="Whole Pipeline")
    # st.markdown(
    #     """ 
    # Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    # """
    #)

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to Forcast the Sales for upcoming Weeks. You can input these feature of your area and information about next week and you get your Forcasted Sales. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Store |  Enter Store number  | 
    | Holiday_Flag   |  If it is a holiday week or not. for holiday week enter 1 otherwise 0. |  
    | CPI	 |       Consumer Price Index    | 
    | Unemployment |       Unemployment Rate in the City.  |
    | month |  Enter month number    | 
    | Quarter	|    Enter Quater of the year Range(1-4). |
    | Week_Number |    Enter Week number of the year. |
    | Seasonal  |    if it is december then enter 2 or if it is november then enter 1 otherwise 0. |
    """
    )

    Store = st.number_input("Store Number")
    Holiday_Flag = st.number_input("Holiday week or not")
    CPI =  st.number_input("Consumer Price Index")
    Unemployment = st.number_input("Unemployment Rate in the City")
    month = st.number_input("Enter month number")
    Quarter	=   st.number_input("Enter Quater of the year Range(1-4)")
    Week_Number = st.number_input("Enter Week number of the year")  
    Seasonal = st.number_input("if it is december then enter 2 or if it is november then enter 1 otherwise 0")
    

    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_main()

        df = pd.DataFrame(
            {
                "Store": [Store],
                "Holiday_Flag": [Holiday_Flag],	
                "CPI": [CPI],  	
                "Unemployment": [Unemployment],	
                "month": [month],	
                "Quarter": [Quarter],
                "Week_Number": [Week_Number],
                "Seasonal": [Seasonal],
                
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your Forcasted Sales for the next week is :-{}".format(
                pred
            )
        )
    # if st.button("Results"):
    #     st.write(
    #         "We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:"
    #     )

    #     df = pd.DataFrame(
    #         {
    #             "Models": ["LightGBM", "Xgboost"],
    #             "MSE": [1.804, 1.781],
    #             "RMSE": [1.343, 1.335],
    #         }
    #     )
    #     st.dataframe(df)

    #     st.write(
    #         "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
    #     )
    #     image = Image.open("_assets/feature_importance_gain.png")
    #     st.image(image, caption="Feature Importance Gain")


if __name__ == "__main__":
    main()