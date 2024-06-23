def calculate_calibration_error(calibration_data, calibration_model):
    # Calculate the error between the calibration data and the calibration model
    error = calibration_data - calibration_model
    return error

