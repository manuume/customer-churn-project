import great_expectations as gx
import pandas as pd
from typing import Tuple, List
from great_expectations.validator.validator import Validator

def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    Updated for GX 1.x compatibility.
    """
    print("🔍 Starting data validation with Great Expectations...")

    # ==========================================
    # 🚨 CRITICAL FIXES FOR DATA QUALITY 🚨
    # ==========================================
    
    # 1. Fix "String vs Number" crash
    # TotalCharges is read as object (string) due to empty spaces. Convert to numeric.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    
    # 2. Close the 'NaN' Loophole
    # 'coerce' creates NaNs for new customers (tenure=0). 
    # We must fill them with 0.0 to avoid breaking Preprocessing later.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # ==========================================
    # 1. SETUP (Updated for GX 1.x)
    # ==========================================
    context = gx.get_context()
    
    datasource_name = "temp_validation_datasource"
    asset_name = "telco_dataframe_asset"
    suite_name = "telco_validation_suite"

    # Clean up existing datasource to prevent errors on re-runs
    if datasource_name in context.data_sources.all():
        context.delete_datasource(datasource_name)

    # Create Datasource and Read DataFrame (returns a Batch)
    datasource = context.data_sources.add_pandas(datasource_name)
    batch = datasource.read_dataframe(df, asset_name=asset_name)

    # Create and Register Expectation Suite
    if suite_name in context.suites.all():
        context.suites.delete(suite_name)
    
    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add(suite)

    # Create Validator manually
    validator = Validator(
        execution_engine=datasource.get_execution_engine(),
        batches=[batch],
        data_context=context,
        expectation_suite=suite
    )
    # ==========================================

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("    Validating schema and required columns...")

    # Customer identifier
    validator.expect_column_to_exist("customerID")
    validator.expect_column_values_to_not_be_null("customerID")

    # Core demographic features
    validator.expect_column_to_exist("gender") 
    validator.expect_column_to_exist("Partner")
    validator.expect_column_to_exist("Dependents")

    # Service features
    validator.expect_column_to_exist("PhoneService")
    validator.expect_column_to_exist("InternetService")
    validator.expect_column_to_exist("Contract")

    # Financial features
    validator.expect_column_to_exist("tenure")
    validator.expect_column_to_exist("MonthlyCharges")
    validator.expect_column_to_exist("TotalCharges")

    # === BUSINESS LOGIC VALIDATION ===
    print("    Validating business logic constraints...")

    # Gender
    validator.expect_column_values_to_be_in_set("gender", ["Male", "Female"])

    # Yes/No fields
    validator.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    validator.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    validator.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])

    # Contract types
    validator.expect_column_values_to_be_in_set(
        "Contract", 
        ["Month-to-month", "One year", "Two year"]
    )

    # Internet service types
    validator.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )

    # === NUMERIC RANGE VALIDATION ===
    print("    Validating numeric ranges and business constraints...")

    # Tenure
    validator.expect_column_values_to_be_between("tenure", min_value=0)

    # Monthly charges
    validator.expect_column_values_to_be_between("MonthlyCharges", min_value=0)

    # Total charges (Safe now because we converted to float)
    validator.expect_column_values_to_be_between("TotalCharges", min_value=0)

    # === STATISTICAL VALIDATION ===
    print("    Validating statistical properties...")

    # Tenure
    validator.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)

    # Monthly charges
    validator.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)

    # No missing values in critical numeric features  
    validator.expect_column_values_to_not_be_null("tenure")
    validator.expect_column_values_to_not_be_null("MonthlyCharges")
    
    # NEW: Now that we filled NaNs, we can enforce this check too
    validator.expect_column_values_to_not_be_null("TotalCharges")

    # === DATA CONSISTENCY CHECKS ===
    print("    Validating data consistency...")

    # Total charges should generally be >= Monthly charges
    validator.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95 
    )

    # === RUN VALIDATION SUITE ===
    print("     Running complete validation suite...")
    results = validator.validate()

    # === PROCESS RESULTS (GX 1.x) ===
    failed_expectations = []
    
    for r in results.results:
        if not r.success:
            expectation_type = r.expectation_config.expectation_type
            failed_expectations.append(expectation_type)

    total_checks = len(results.results)
    passed_checks = total_checks - len(failed_expectations)

    if results.success:
        print(f" Data validation PASSED: {passed_checks}/{total_checks} checks successful")
        return True, []
    else:
        print(f" Data validation FAILED: {len(failed_expectations)}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
        return False, failed_expectations
