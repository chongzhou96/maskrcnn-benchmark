# -port : The port to run the server on.
# -hostname : The hostname to run the server on.
# -base_url : The base server url (default = /).
# -env_path : The path to the serialized session to reload.
# -logging_level : Logging level (default = INFO). Accepts both standard text and numeric logging values.
# -readonly : Flag to start server in readonly mode.
# -enable_login : Flag to setup authentication for the sever, requiring a username and password to login.
# -force_new_cookie : Flag to reset the secure cookie used by the server, 
#                       invalidating current login cookies. Requires -enable_login.

visdom -port 8081