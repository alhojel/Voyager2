from voyager import Voyager

# You can also use mc_port instead of azure_login, but azure_login is highly recommended
azure_login = {
    "client_id": "3ba45205-c9c0-46bc-8216-4b481f55873d",
    "redirect_url": "https://127.0.0.1/auth-response",
    "secret_value": "",
    "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
}
openai_api_key = ""

voyager = Voyager(
    azure_login=azure_login,
    openai_api_key=openai_api_key,
    ckpt_dir="ckpt2",
    pause_on_think=False,
)

# start lifelong learning
voyager.learn()