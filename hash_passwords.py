import streamlit_authenticator as stauth

# Generate hashed passwords
hashed_passwords = stauth.Hasher(['admin123', 'rapas456']).generate()

print("Generated hashed passwords:")
print(hashed_passwords)
print("\nReplace the placeholder passwords in config.yaml with these "
      "hashed values")
