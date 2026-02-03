## Database Setup Steps (PostgreSQL)
### Install PostgreSQL

``
sudo apt install postgresql postgresql-contrib
``
### Ensure its running
``
sudo systemctl start postgresql
``
### Install Python dependencies
``
pip install -r requirements.txt
``

### Run the setupDB script
``
sudo -u postgres python setupDB.py
``
###
This creacte cofina_db, create a user 'cofina_user' with user password 'strongpassword'. It grant required schema privileges Create all required tables: users, user_preferences and agent_decisions_log

