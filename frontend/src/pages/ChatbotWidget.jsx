import React, { useState } from "react";
import { FaComments, FaTimes } from "react-icons/fa";

export default function ChatbotWidget() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div
      style={{
        position: "fixed",
        bottom: "24px",
        right: "24px",
        zIndex: 9999,
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
        gap: "12px",
      }}
    >
      {/* The Chat Window */}
      {isOpen && (
        <div
          style={{
            width: "360px",
            height: "500px",
            boxShadow: "0 8px 32px rgba(0,0,0,0.25)",
            borderRadius: "16px",
            overflow: "hidden",
            background: "#fff",
            display: "flex",
            flexDirection: "column",
            border: "1px solid rgba(0,0,0,0.1)",
            animation: "slideUp 0.3s ease-out",
          }}
        >
          {/* Header */}
          <div style={{
            background: "#2563eb",
            color: "white",
            padding: "12px 16px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontWeight: "bold",
          }}>
            <span className="flex items-center gap-2">🤖 AI assistant</span>
            <button 
              onClick={() => setIsOpen(false)}
              style={{ background: "transparent", border: "none", color: "white", cursor: "pointer" }}
            >
              <FaTimes size={18} />
            </button>
          </div>

          <iframe
            title="ecg-dialogflow-bot"
            src="https://console.dialogflow.com/api-client/demo/embedded/2aa3370d-1ac1-41c0-a3c7-2b6d29da93a2"
            style={{
              width: "100%",
              flexGrow: 1,
              border: "none",
            }}
            allow="microphone;"
          ></iframe>
        </div>
      )}

      {/* Floating Toggle Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          style={{
            width: "60px",
            height: "60px",
            borderRadius: "30px",
            background: "#2563eb",
            color: "white",
            border: "none",
            cursor: "pointer",
            boxShadow: "0 4px 12px rgba(37, 99, 235, 0.4)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
          }}
          onMouseEnter={(e) => e.currentTarget.style.transform = "scale(1.1)"}
          onMouseLeave={(e) => e.currentTarget.style.transform = "scale(1)"}
        >
          <FaComments size={28} />
        </button>
      )}

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes slideUp {
          from { transform: translateY(20px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
      `}} />
    </div>
  );
}
