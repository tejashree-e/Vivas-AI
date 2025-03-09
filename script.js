document.addEventListener("DOMContentLoaded", function () {
  const chatBox = document.getElementById("chat-box");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const switchLangBtn = document.getElementById("switch-lang");
  const quickQueryButtons = document.querySelectorAll(".quick-query");
  
  let lang = "en"; // Default language

  function appendMessage(sender, message) {
      const div = document.createElement("div");
      div.classList.add("message");
      div.textContent = `${sender}: ${message}`;
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
  }

  function fetchResponse(message) {
      navigator.geolocation.getCurrentPosition(position => {
          const location = {
              latitude: position.coords.latitude,
              longitude: position.coords.longitude
          };

          fetch("http://127.0.0.1:5000/chat", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message, language: lang, location })
          })
          .then(response => response.json())
          .then(data => appendMessage("Bot", data.response))
          .catch(() => appendMessage("Bot", "Error fetching response."));
      });
  }

  sendBtn.addEventListener("click", function () {
      const message = userInput.value.trim();
      if (message) {
          appendMessage("You", message);
          fetchResponse(message);
          userInput.value = "";
      }
  });

  userInput.addEventListener("keypress", function (e) {
      if (e.key === "Enter") sendBtn.click();
  });

  switchLangBtn.addEventListener("click", function () {
      lang = lang === "en" ? "ta" : "en";
      switchLangBtn.textContent = lang === "en" ? "Switch to Tamil" : "ஆங்கிலத்திற்கு மாற்று";
  });

  quickQueryButtons.forEach(button => {
      button.addEventListener("click", function () {
          userInput.value = button.dataset.query;
          sendBtn.click();
      });
  });
});