import pandas as pd
import pygsheets
from datetime import date
import json

def save_feedback_to_sheets(data):
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Authenticate and access the Google Sheets document
        client = pygsheets.authorize(service_account_file="feedback.json")
        spreadsheet_name = "trackers"
        worksheet_title = str(date.today())

        # Check if the Google Sheets document exists, or create it if it doesn't
        try:
            spreadsheet = client.open(spreadsheet_name)
        except pygsheets.SpreadsheetNotFound:
            # Create a new Google Sheets document if it doesn't exist
            spreadsheet = client.create(spreadsheet_name)

        # Check if the worksheet with the specified title exists, and create it if it doesn't
        try:
            worksheet = spreadsheet.worksheet_by_title(worksheet_title)
        except pygsheets.exceptions.WorksheetNotFound:
            # Create a new worksheet with the specified title
            worksheet = spreadsheet.add_worksheet(worksheet_title)
            spreadsheet_url = spreadsheet.url
            logging.info("URL of the Google Sheets document: %s", spreadsheet_url)

        # Load the existing data from the worksheet into a DataFrame, or create a new DataFrame if the worksheet is empty
        try:
            existing_data = worksheet.get_as_df(has_header=True)
        except pygsheets.exceptions.WorksheetNotFound:
            existing_data = pd.DataFrame(columns=["Item1", "Item2", "Item3", "Item4"])

        # Define the data for a new row
        new_data = json.loads(data)

        # Append the new row to the DataFrame
        new_row = pd.DataFrame([new_data])
        logging.info("New Data: %s", new_data)
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)

        # Update the Google Sheets worksheet with the updated data
        worksheet.set_dataframe(updated_data, start="A1")

    except Exception as e:
        logging.error("An error occurred: %s", str(e))