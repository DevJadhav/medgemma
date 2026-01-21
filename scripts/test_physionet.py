#!/usr/bin/env python3
"""Test PhysioNet credentials."""

import requests
from requests.auth import HTTPBasicAuth
import sys

def test_credentials(username: str, password: str):
    """Test PhysioNet authentication."""
    print(f'Testing credentials for user: {username}')
    
    test_url = 'https://physionet.org/files/mimiciv/3.1/README'
    auth = HTTPBasicAuth(username, password)
    
    try:
        response = requests.get(test_url, auth=auth, timeout=30)
        print(f'Status code: {response.status_code}')
        
        if response.status_code == 200:
            print('✅ Authentication successful!')
            print(f'Response preview: {response.text[:300]}...')
            return True
        elif response.status_code == 401:
            print('❌ Authentication failed - incorrect credentials')
        elif response.status_code == 403:
            print('❌ Access forbidden - credentials may not have access to MIMIC-IV')
            print('Make sure you have completed the CITI training and signed the data use agreement.')
        else:
            print(f'Unexpected status. Response: {response.text[:500]}')
        return False
    except Exception as e:
        print(f'Error: {e}')
        return False

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python test_physionet.py <username> <password>")
        sys.exit(1)
    
    username = sys.argv[1]
    password = sys.argv[2]
    success = test_credentials(username, password)
    sys.exit(0 if success else 1)
