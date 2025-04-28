package fr.istic.web.rest;

import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * REST controller to process PDF uploads and execute a Python script
 */
@Path("/api")
public class PythonControllerES {

    private final Logger log = LoggerFactory.getLogger(PythonControllerES.class);

    
    @POST
    @Path("/add-pdf")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response processPdfAndRunScript(Map<String, Object> requestData) {
        Map<String, Object> response = new HashMap<>();
        StringBuilder output = new StringBuilder();
        StringBuilder errorOutput = new StringBuilder();
    
        try {
            log.info("Starting the process to add PDF to Elasticsearch...");
    
            // Retrieve and validate the base64-encoded PDF data
            String base64Data = requestData.containsKey("pdfData") ? requestData.get("pdfData").toString() : "";
            if (base64Data.isEmpty()) {
                log.error("No PDF data provided in the request.");
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "No PDF data provided in the request."))
                        .build();
            }
    
            // Retrieve courseName
            String courseName = requestData.containsKey("courseName") ? requestData.get("courseName").toString() : "";
            if (courseName.isEmpty()) {
                log.error("Course name missing in the request.");
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "Course name is missing."))
                        .build();
            }
    
            // Retrieve PDF name
            String pdfName = requestData.containsKey("pdfName") ? requestData.get("pdfName").toString() : "";
            if (pdfName.isEmpty()) {
                log.error("PDF name missing in the request.");
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "PDF name is missing."))
                        .build();
            }
    
            // Decode base64 and save it as a temporary PDF file
            byte[] pdfBytes = Base64.getDecoder().decode(base64Data);
            String tempPdfPath = "/tmp/" + pdfName;
            Files.write(Paths.get(tempPdfPath), pdfBytes);
            log.info("PDF file saved at: " + tempPdfPath);
    
            // Path to Python script
            String scriptPath = "src/main/resources/rag/add_pdf_to_es.py";
    
            // Check if the script file exists
            File scriptFile = new File(scriptPath);
            if (!scriptFile.exists()) {
                log.error("Python script not found: " + scriptPath);
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "Python script not found: " + scriptPath))
                        .build();
            }
    
            // Run the Python script with arguments: tempPdfPath, courseName, and pdfName
            ProcessBuilder pb = new ProcessBuilder(
                    "python3",
                    scriptFile.getAbsolutePath(),
                    tempPdfPath,
                    courseName,
                    pdfName
            );
            pb.directory(scriptFile.getParentFile()); // Set working directory
            Process process = pb.start();
    
            // Read the Python script's standard output
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                log.info("Python Output: " + line);
            }
    
            // Read any errors from the Python script
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String errorLine;
            while ((errorLine = errorReader.readLine()) != null) {
                errorOutput.append(errorLine).append("\n");
                log.error("Python Error/Warning: " + errorLine);
            }
    
            // Wait for the process to finish
            int exitCode = process.waitFor();
            log.info("Process finished with exit code: " + exitCode);
    
            // Build the response JSON
            response.put("exitCode", exitCode);
            response.put("output", output.toString());
    
            if (exitCode == 0) {
                // Include script output in the response
                response.put("status", "success");
                response.put("message", "PDF data added to Elasticsearch successfully.");
    
                if (errorOutput.length() > 0) {
                    response.put("warnings", errorOutput.toString());
                }
                return Response.ok(response).build();
            } else {
                // Include error details in the response
                response.put("status", "failure");
                response.put("error", errorOutput.toString());
                return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                        .entity(response)
                        .build();
            }
    
        } catch (Exception e) {
            log.error("Error while processing the PDF and running the Python script.", e);
            response.put("error", "Error while processing the PDF: " + e.getMessage());
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(response)
                    .build();
        }
    }
    

    @POST
    @Path("/delete-chunks")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response deleteChunks(Map<String, Object> requestData) {
        Map<String, Object> response = new HashMap<>();
        StringBuilder output = new StringBuilder();
        StringBuilder errorOutput = new StringBuilder();

        try {
            log.info("Starting the process to delete chunks...");

            // Retrieve and validate the course name
            String courseName = requestData.containsKey("courseName") ? requestData.get("courseName").toString() : "";
            if (courseName.isEmpty()) {
                log.error("Course name missing in the request.");
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "Course name is missing."))
                        .build();
            }

            // Retrieve and validate the PDF name (optional)
            String pdfName = requestData.containsKey("pdfName") ? requestData.get("pdfName").toString() : "";

            // Path to Python script
            String scriptPath = "src/main/resources/rag/delete_chunks_from_es.py";

            // Check if the script file exists
            File scriptFile = new File(scriptPath);
            if (!scriptFile.exists()) {
                log.error("Python script not found: " + scriptPath);
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "Python script not found: " + scriptPath))
                        .build();
            }

            // Run the Python script with the course name and optional PDF name as arguments
            ProcessBuilder pb = new ProcessBuilder(
                    "python3",
                    scriptFile.getAbsolutePath(),
                    courseName,
                    pdfName
            );
            pb.directory(scriptFile.getParentFile()); // Set working directory
            Process process = pb.start();

            // Read the Python script's standard output
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                log.info("Python Output: " + line);
            }

            // Read any errors from the Python script
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String errorLine;
            while ((errorLine = errorReader.readLine()) != null) {
                errorOutput.append(errorLine).append("\n");
                log.error("Python Error/Warning: " + errorLine);
            }

            // Wait for the process to finish
            int exitCode = process.waitFor();
            log.info("Process finished with exit code: " + exitCode);

            // Build the response JSON
            response.put("exitCode", exitCode);
            response.put("output", output.toString());

            if (exitCode == 0) {
                response.put("status", "success");
                response.put("message", "Chunks deleted successfully.");
                if (errorOutput.length() > 0) {
                    response.put("warnings", errorOutput.toString());
                }
                return Response.ok(response).build();
            } else {
                response.put("status", "failure");
                response.put("error", errorOutput.toString());
                return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                        .entity(response)
                        .build();
            }

        } catch (Exception e) {
            log.error("Error while deleting chunks.", e);
            response.put("error", "Error while deleting chunks: " + e.getMessage());
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(response)
                    .build();
        }
    }

    @POST
    @Path("/get-pdf-names")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response getPdfNames(Map<String, Object> requestData) {
        Map<String, Object> response = new HashMap<>();
        StringBuilder output = new StringBuilder();
        StringBuilder errorOutput = new StringBuilder();

        try {
            log.info("Starting the process to get all PDF names...");

            // Retrieve courseName from the request
            String courseName = requestData.containsKey("courseName") ? requestData.get("courseName").toString() : "";
            if (courseName.isEmpty()) {
                log.error("Course name missing in the request.");
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "Course name is missing."))
                        .build();
            }

            // Path to Python script
            String scriptPath = "src/main/resources/rag/get_list_of_pdf.py";

            // Check if script file exists
            File scriptFile = new File(scriptPath);
            if (!scriptFile.exists()) {
                log.error("Python script not found: " + scriptPath);
                return Response.status(Response.Status.BAD_REQUEST)
                        .entity(Map.of("error", "Python script not found: " + scriptPath))
                        .build();
            }

            // Run Python script with courseName as argument
            ProcessBuilder pb = new ProcessBuilder(
                    "python3",
                    scriptFile.getAbsolutePath(),
                    courseName
            );
            pb.directory(scriptFile.getParentFile()); // Set working directory
            Process process = pb.start();

            // Read the Python script's standard output
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                log.info("Python Output: " + line);
            }

            // Read any errors from the Python script
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String errorLine;
            while ((errorLine = errorReader.readLine()) != null) {
                errorOutput.append(errorLine).append("\n");
                log.error("Python Error/Warning: " + errorLine);
            }

            // Wait for the process to finish
            int exitCode = process.waitFor();
            log.info("Process finished with exit code: " + exitCode);

            // Build the response JSON
            response.put("exitCode", exitCode);
            response.put("output", output.toString());

            if (exitCode == 0) {
                response.put("status", "success");
                response.put("message", "PDF names retrieved successfully.");
                if (errorOutput.length() > 0) {
                    response.put("warnings", errorOutput.toString());
                }
                return Response.ok(response).build();
            } else {
                response.put("status", "failure");
                response.put("error", errorOutput.toString());
                return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                        .entity(response)
                        .build();
            }

        } catch (Exception e) {
            log.error("Error while retrieving PDF names.", e);
            response.put("error", "Error while retrieving PDF names: " + e.getMessage());
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity(response)
                    .build();
        }
    }


    
}
