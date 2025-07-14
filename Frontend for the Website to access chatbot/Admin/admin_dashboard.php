<?php
session_start();
if (!isset($_SESSION['admin'])) {
    header("Location: admin_login.php");
    exit();
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f4; }
        .sidebar { width: 250px; height: 100vh; background: #333; color: white; position: fixed; top: 0; left: 0; padding: 20px; }
        .sidebar h2 { text-align: center; margin-bottom: 30px; }
        .sidebar ul { list-style: none; padding: 0; }
        .sidebar ul li { margin: 20px 0; }
        .sidebar ul li a { color: white; text-decoration: none; font-size: 18px; display: flex; align-items: center; }
        .sidebar ul li a i { margin-right: 10px; }
        .sidebar ul li a:hover { background: #575757; padding-left: 10px; transition: 0.3s; }
        .main-content { margin-left: 270px; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; background: #007bff; color: white; padding: 10px 20px; }
        .header h1 { margin: 0; }
        .header .logout { background: #ff4d4d; padding: 10px 20px; border: none; cursor: pointer; color: white; }
        .cards { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); width: 30%; text-align: center; }
        .card h3 { margin-bottom: 20px; }
        .card a { display: inline-block; margin-top: 10px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; transition: background 0.3s; }
        .card a:hover { background: #0056b3; }
    </style>
</head>
<body>

<div class="sidebar">
    <h2>Admin Panel</h2>
    <ul>
        <li><a href="admin_dashboard.php"><i class="fas fa-tachometer-alt"></i> Dashboard</a></li>
        <li><a href="manage_doctors.php"><i class="fas fa-user-md"></i> Manage Doctors</a></li>
        <li><a href="add_doctor.php"><i class="fas fa-user-plus"></i> Add Doctor</a></li>
        <li><a href="manage_appointments.php"><i class="fas fa-calendar-alt"></i> Manage Appointments</a></li>
        <li><a href="secure_password_reset.php"><i class="fas fa-lock"></i> Secure Password Reset</a></li>
        <li><a href="logout.php"><i class="fas fa-sign-out-alt"></i> Logout</a></li>
    </ul>
</div>

<div class="main-content">
    <div class="header">
        <h1>Welcome, Admin</h1>
        <button class="logout" onclick="window.location.href='logout.php'">Logout</button>
    </div>
    <div class="cards">
        <div class="card">
            <h3>Manage Doctors</h3>
            <p>View and manage the list of doctors.</p>
            <a href="manage_doctors.php">Go to Doctors</a>
        </div>
        <div class="card">
            <h3>Add Doctor</h3>
            <p>Add a new doctor to the system.</p>
            <a href="add_doctor.php">Add Doctor</a>
        </div>
        <div class="card">
            <h3>Manage Appointments</h3>
            <p>View and manage appointments.</p>
            <a href="manage_appointments.php">Go to Appointments</a>
        </div>
        <div class="card">
            <h3>Secure Password Reset</h3>
            <p>Reset passwords securely.</p>
            <a href="secure_password_reset.php">Go to Password Reset</a>
        </div>
    </div>
</div>

<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
