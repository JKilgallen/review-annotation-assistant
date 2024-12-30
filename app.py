import streamlit as st
import pandas as pd
import os

# -------------------------------------------------------------------
# Configurations & Filenames
# -------------------------------------------------------------------
LABELED_DATA_FILE = "temp_labeled_data.csv"
INDEX_FILE = "temp_index.txt"

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def load_csv(uploaded_file) -> pd.DataFrame:
    """Load an uploaded CSV file into a Pandas DataFrame."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.DataFrame()

    return df

def load_labeled_data() -> pd.DataFrame:
    """
    Load existing labeled data from disk if it exists;
    otherwise return an empty DataFrame.
    """
    if os.path.exists(LABELED_DATA_FILE):
        return pd.read_csv(LABELED_DATA_FILE)
    else:
        return pd.DataFrame(columns=["review_id", "review_text", "label_id", "sentiment"])

def save_labeled_data(df: pd.DataFrame):
    """
    Save labeled data DataFrame to disk as CSV, overwriting any existing file.
    """
    df.to_csv(LABELED_DATA_FILE, index=False)

def append_labeled_data(new_records: list):
    """
    Append new labeling records to the labeled data CSV on disk.
    Each record is a dict with keys [review_id, review_text, label_id, sentiment].
    """
    existing_df = load_labeled_data()  # load existing labeled data
    new_df = pd.DataFrame(new_records)  # convert new records to DataFrame
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    save_labeled_data(combined)

def load_current_index() -> int:
    """
    Load the current review index from a text file.
    Return 0 if the file does not exist or is invalid.
    """
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "r") as f:
                idx_str = f.read().strip()
                return int(idx_str)
        except:
            return 0
    else:
        return 0

def save_current_index(index: int):
    """
    Save the current review index to a text file.
    """
    with open(INDEX_FILE, "w") as f:
        f.write(str(index))

def clear_label_checkboxes(labels: list):
    """
    Uncheck checkboxes and reset sentiment dropdowns so that
    after moving to another record, the next record starts fresh.
    """
    for label in labels:
        st.session_state[f"checkbox_{label}"] = False
        st.session_state[f"sentiment_{label}"] = "Neutral"

# -------------------------------------------------------------------
# Page Functions
# -------------------------------------------------------------------
def home_page():
    """Page 1: Allows the user to upload review data and label definitions."""
    st.title("Review & Label Setup")

    st.write("## 1. Upload Review Data")
    text_file = st.file_uploader(
        label="Upload a CSV file containing your review text data",
        type=["csv"],
        key="text_file_uploader",
    )
    if text_file:
        st.session_state.text_df = load_csv(text_file)
        st.write("### Preview of Text Data")
        st.dataframe(st.session_state.text_df.head(), use_container_width=True)
        if not all(col in st.session_state.text_df.columns for col in ["review_id", "review_text"]):
            st.error("Your text CSV must have 'review_id' and 'review_text' columns.")

    st.write("---")
    st.write("## 2. Upload Label Definitions")
    label_file = st.file_uploader(
        label="Upload a CSV file containing label definitions",
        type=["csv"],
        key="label_file_uploader",
    )
    if label_file:
        st.session_state.labels_df = load_csv(label_file)
        st.write("### Preview of Label Definitions")
        st.dataframe(st.session_state.labels_df.head(), use_container_width=True)
        if not all(col in st.session_state.labels_df.columns for col in ["label_id", "label_definition"]):
            st.error("Your labels CSV must have 'label_id' and 'label_definition' columns.")

    st.write("---")

    # Proceed button
    can_proceed = (
        not st.session_state.text_df.empty
        and not st.session_state.labels_df.empty
        and all(col in st.session_state.text_df.columns for col in ["review_id", "review_text"])
        and all(col in st.session_state.labels_df.columns for col in ["label_id", "label_definition"])
    )

    if st.button("Proceed to Labeling", disabled=not can_proceed):
        st.session_state.labeled_data_df = load_labeled_data()
        st.session_state.current_index = load_current_index()
        st.session_state.page = "label"
        st.rerun()

def label_page():
    """Page 2: Main labeling interface with slider for sentiment and no in-line label definitions, using a form."""
    st.title("Label Your Reviews")

    # 1) Safety checks
    if st.session_state.text_df.empty or st.session_state.labels_df.empty:
        st.warning("Please upload text data and label definitions first.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    total_reviews = len(st.session_state.text_df)
    if total_reviews == 0:
        st.warning("No reviews found in the uploaded data.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    # 3) Validate current index
    if st.session_state.current_index >= total_reviews:
        st.session_state.current_index = 0

    # 4) Display current review
    current_row = st.session_state.text_df.iloc[st.session_state.current_index]
    review_id = current_row["review_id"]
    review_text = current_row["review_text"]

    st.markdown(f"**Review {st.session_state.current_index + 1} of {total_reviews}**")
    st.write("---")
    st.subheader(f"**Review ID:** {review_id}")
    st.markdown("""**Review Text:**""")
    # display words in label ids in different color
    for label_id, _ in st.session_state.labels_df.values:
        review_text = review_text.replace(label_id, f":green-background[{label_id}]")
    st.markdown(review_text)


    # 5) Optional: Quick reference in sidebar
    with st.sidebar:
        st.title("Quick Reference")
        st.write("Label Definitions")
        st.table(st.session_state.labels_df.set_index("label_id"))

    # 6) The labeling form
    st.write("### Assign Labels & Sentiments")
    labels = st.session_state.labels_df["label_id"].tolist()

    with st.form("labeling_form", clear_on_submit=True):
        def enable_slider(slider_key):
            st.session_state[slider_key].disabled = not st.session_state[checkbox_key]

        n_cols = 3
        columns = {i: col for i, col in enumerate(st.columns(n_cols))}
        # Create a checkbox + slider pair for each label
        for idx, label in enumerate(labels):
            with columns[idx%n_cols]:
                checkbox_key = f"checkbox_{label}"
                slider_key = f"slider_{label}"

                selected = st.checkbox(label, key=checkbox_key)
                sentiment = st.select_slider(
                    "",
                    options=["Negative", "Neutral", "Positive"],
                    value="Neutral",  # default
                    key=slider_key
                )

        # -- Form Buttons --
        st.write("---")
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1,1,1,3])
        with nav_col1:
            previous_btn = st.form_submit_button("Previous")
        with nav_col2:
            save_next_btn = st.form_submit_button("Save & Next")
        with nav_col3:
            done_btn = st.form_submit_button("Done Labeling")
        with nav_col4:
            home_btn = st.form_submit_button("Back to Home")

    # 7) After form submission, figure out which button was clicked:
    if previous_btn:
        # Move to the previous review
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1
        save_current_index(st.session_state.current_index)

        st.rerun()

    elif save_next_btn:
        # For each ticked checkbox, save the label and sentiment
        selected_labels_sentiments = {}
        for label in labels:
            checkbox_key = f"checkbox_{label}"
            slider_key = f"slider_{label}"
            if st.session_state[checkbox_key]:
                selected_labels_sentiments[label] = st.session_state[slider_key]

        new_records = []
        for label in labels:
            if st.session_state[f"checkbox_{label}"]:
                new_records.append({
                    "review_id": review_id,
                    "review_text": review_text,
                    "label_id": label,
                    "sentiment": st.session_state[f"slider_{label}"],
                })
        if new_records:
            append_labeled_data(new_records)

        # Move to next review
        st.session_state.current_index += 1
        save_current_index(st.session_state.current_index)
        st.rerun()

    elif done_btn:
        # Go to the export page
        st.session_state.page = "export"
        st.rerun()

    elif home_btn:
        # Go back to home
        st.session_state.page = "home"
        st.rerun()



def export_page():
    """Page 3: Allows downloading the labeled data."""
    st.title("Export Labeled Data")

    labeled_df = load_labeled_data()
    if labeled_df.empty:
        st.warning("No labeled data yet. Label some reviews first.")
        if st.button("Back to Labeling"):
            st.session_state.page = "label"
            st.rerun()
        return

    st.write("Total labeled entries:", len(labeled_df))
    st.write("Preview of labeled data:")
    st.dataframe(labeled_df.tail(10), use_container_width=True)

    csv_data = labeled_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="labeled_data.csv",
        mime="text/csv",
    )

    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()

# -------------------------------------------------------------------
# Main App
# -------------------------------------------------------------------
def main():
    # Initialize session state variables
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "text_df" not in st.session_state:
        st.session_state.text_df = pd.DataFrame()
    if "labels_df" not in st.session_state:
        st.session_state.labels_df = pd.DataFrame()
    if "labeled_data_df" not in st.session_state:
        st.session_state.labeled_data_df = pd.DataFrame()

    # Routing
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "label":
        label_page()
    elif st.session_state.page == "export":
        export_page()

if __name__ == "__main__":
    main()
