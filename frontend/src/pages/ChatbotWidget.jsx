import React from "react";

export default function ChatbotWidget() {
  return (
    <div
      style={{
        position: "fixed",
        bottom: "24px",
        right: "24px",
        width: "360px",
        height: "500px",
        zIndex: 9999,
        boxShadow: "0 4px 20px rgba(0,0,0,0.2)",
        borderRadius: "12px",
        overflow: "hidden",
        background: "#fff",
      }}
    >
      <iframe
        title="ecg-dialogflow-bot"
        src="https://console.dialogflow.com/api-client/demo/embedded/2aa3370d-1ac1-41c0-a3c7-2b6d29da93a2"
        style={{
          width: "100%",
          height: "100%",
          border: "none",
        }}
        allow="microphone;"
      ></iframe>
    </div>
  );
}
