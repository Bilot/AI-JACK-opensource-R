
#* @param param parameter string values
#* @param param parameter string colnames
#* @post /predict
function(param, param2, param3){
    # Parse data.frame:
    df <- create_df(param, param2, param3)
    # Predict:
    plumber_predict(df, set, param, param2, param3, "")
}
    