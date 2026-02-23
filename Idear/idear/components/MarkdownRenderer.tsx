"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

interface MarkdownRendererProps {
  initialContent?: string;
}

export default function MarkdownRenderer({
  initialContent = "",
}: MarkdownRendererProps) {
  const [isEditing, setIsEditing] = useState(true);
  const [content, setContent] = useState(initialContent);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const switchToRender = useCallback(() => {
    setIsEditing(false);
  }, []);

  const switchToEdit = useCallback(() => {
    setIsEditing(true);
  }, []);

  const toggleMode = useCallback(() => {
    setIsEditing((prev) => !prev);
  }, []);

  // Ctrl + Shift + V to switch to render mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "v") {
        e.preventDefault();
        switchToRender();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [switchToRender]);

  // Focus the textarea when switching to edit mode
  useEffect(() => {
    if (isEditing && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isEditing]);

  return (
    <div className="markdown-renderer">
      <div className="toolbar">
        <button onClick={toggleMode}>
          {isEditing ? "Render" : "Edit"}
        </button>
      </div>

      {isEditing ? (
        <textarea
          ref={textareaRef}
          className="markdown-editor"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Write your markdown here..."
        />
      ) : (
        <div
          className="markdown-preview"
          onDoubleClick={switchToEdit}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              switchToEdit();
            }
          }}
          role="button"
          tabIndex={0}
          title="Double-click or press Enter to edit"
        >
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}
