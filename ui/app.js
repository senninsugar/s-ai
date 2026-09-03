const chat = document.getElementById("chat");
const input = document.getElementById("input");
const send = document.getElementById("send");

let messages = [];

function addMessage(role, text) {

    const wrapper = document.createElement("div");

    wrapper.className =
        `message ${role}`;

    const avatar = document.createElement("div");

    avatar.className = "avatar";

    avatar.textContent =
        role === "user"
            ? "U"
            : "N";

    const bubble = document.createElement("div");

    bubble.className = "bubble";

    bubble.textContent = text;

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);

    chat.appendChild(wrapper);

    chat.scrollTop = chat.scrollHeight;

    return bubble;
}

async function sendMessage() {

    const text = input.value.trim();

    if (!text) {
        return;
    }

    input.value = "";

    input.style.height = "42px";

    send.disabled = true;

    messages.push({
        role: "user",
        content: text
    });

    addMessage("user", text);

    const thinking = addMessage(
        "assistant",
        "考えています..."
    );

    try {

        const response = await fetch(
            "/v1/chat/completions",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({

                    model: "novallm",

                    messages: messages,

                    temperature: 0.7,

                    max_tokens: 512

                })
            }
        );

        if (!response.ok) {

            const error =
                await response.text();

            throw new Error(error);
        }

        const data =
            await response.json();

        const answer =
            data.choices?.[0]?.message?.content
            ?? "回答を取得できませんでした。";

        thinking.textContent = answer;

        messages.push({
            role: "assistant",
            content: answer
        });

    } catch (error) {

        console.error(error);

        thinking.textContent =
            "エラーが発生しました。";

    } finally {

        send.disabled = false;

        input.focus();
    }
}

send.addEventListener(
    "click",
    sendMessage
);

input.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }

    }
);

input.addEventListener(
    "input",
    () => {

        input.style.height = "42px";

        input.style.height =
            Math.min(
                input.scrollHeight,
                180
            ) + "px";

    }
);
