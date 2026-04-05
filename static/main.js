let attendanceData = [];
let chartInstance = null;

async function login(event) {
  event.preventDefault();

  const sidInput = document.getElementById("usernameInput");
  const pwdInput = document.getElementById("passwordInput");
  const msg = document.getElementById("msg");
  const submitButton = event.target.querySelector("button[type='submit']");

  const sid = (sidInput?.value || "").trim();
  const pwd = pwdInput?.value || "";

  if (!sid || !pwd) {
    if (msg) {
      msg.innerText = "Please enter both ID and password";
      msg.style.color = "red";
    }
    return;
  }

  try {
    if (submitButton) submitButton.disabled = true;

    const res = await fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sid, pwd }),
      credentials: "same-origin",
    });

    const data = await res.json();

    if (data.success) {
      window.location.href = "/dashboard";
      return;
    }

    if (msg) {
      msg.innerText = data.error || "Invalid login";
      msg.style.color = "red";
    }
  } catch (_err) {
    if (msg) {
      msg.innerText = "Server error";
      msg.style.color = "red";
    }
  } finally {
    if (submitButton) submitButton.disabled = false;
  }
}

async function loadAttendance() {
  try {
    const res = await fetch("/attendance-data", { credentials: "same-origin" });
    if (!res.ok) {
      throw new Error(`Request failed: ${res.status}`);
    }

    attendanceData = await res.json();
    renderTable();
    renderChart();
  } catch (_err) {
    attendanceData = [];
    renderTable();
    renderChart();
  }
}

function renderTable() {
  const table = document.getElementById("attendanceTable");
  if (!table) return;

  if (!attendanceData.length) {
    table.innerHTML = '<tr><td colspan="4">No attendance records found.</td></tr>';
    return;
  }

  const rows = attendanceData
    .map((row) => {
      const statusClass = row.status === "Present" ? "present" : "absent";
      const validIcon = row.valid === 1 ? "&#10003;" : "";
      return `
        <tr>
          <td>${row.date}</td>
          <td>${row.session}</td>
          <td class="${statusClass}">${row.status}</td>
          <td>${validIcon}</td>
        </tr>
      `;
    })
    .join("");

  table.innerHTML = rows;
}

function renderChart() {
  const canvas = document.getElementById("attendanceChart");
  if (!canvas || typeof Chart === "undefined") return;

  const total = attendanceData.length;
  const presentDays = attendanceData.reduce(
    (count, item) => count + (item.status === "Present" ? 1 : 0),
    0
  );
  const absentDays = total - presentDays;

  const presentPercentEl = document.getElementById("presentPercent");
  const absentPercentEl = document.getElementById("absentPercent");

  if (presentPercentEl) {
    presentPercentEl.innerText = total ? `${Math.round((presentDays / total) * 100)}%` : "0%";
  }
  if (absentPercentEl) {
    absentPercentEl.innerText = total ? `${Math.round((absentDays / total) * 100)}%` : "0%";
  }

  if (chartInstance) {
    chartInstance.destroy();
  }

  chartInstance = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["Present", "Absent"],
      datasets: [
        {
          data: [presentDays, absentDays],
          backgroundColor: ["#F59E0B", "#CBD5E1"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      cutout: "60%",
      plugins: {
        legend: { position: "bottom" },
      },
    },
  });
}

async function logout() {
  try {
    await fetch("/logout", { method: "POST", credentials: "same-origin" });
  } catch (_err) {
    // Best-effort logout on server; always reset client view.
  }
  window.location.href = "/";
}

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("attendanceTable")) {
    loadAttendance();
  }
});
