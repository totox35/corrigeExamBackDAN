package fr.istic.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import io.quarkus.hibernate.orm.panache.PanacheQuery;
import io.quarkus.panache.common.Parameters;
import jakarta.json.bind.annotation.JsonbTransient;
import io.quarkus.runtime.annotations.RegisterForReflection;

import jakarta.persistence.*;
import java.io.Serializable;
import java.util.List;
import java.util.Set;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.ArrayList;

// Add these imports for JSON handling
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import java.io.IOException;

/**
 * A Response Group.
 */
@Entity
@Table(name = "response_group")
@RegisterForReflection
public class ResponseGroup extends PanacheEntityBase implements Serializable {

    private static final long serialVersionUID = 1L;
    private static final ObjectMapper objectMapper = new ObjectMapper();

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    public Long id;

    @ManyToOne
    @JoinColumn(name = "question_id")
    @JsonbTransient
    public Question question;

    // Store as JSON string in database
    @Column(name="prediction_ids")
    private String predictionIdsJson;

    // Store as JSON string in database
    @Column(name="average_embedding")
    private String averageEmbeddingJson;

    // Transient fields for use in Java code
    @Transient
    public Long[] predictionIds;

    @Transient
    public Double[] averageEmbedding;

    // Convert JSON to arrays when loading from database
    @PostLoad
    void onLoad() {
        try {
            if (predictionIdsJson != null && !predictionIdsJson.isEmpty()) {
                if (predictionIdsJson.startsWith("[")) {
                    // It's already JSON format
                    predictionIds = objectMapper.readValue(predictionIdsJson, Long[].class);
                } else {
                    // It might be a comma-separated string
                    predictionIds = Arrays.stream(predictionIdsJson.split(","))
                        .map(String::trim)
                        .filter(s -> !s.isEmpty())
                        .map(Long::parseLong)
                        .toArray(Long[]::new);
                }
            } else {
                predictionIds = new Long[0];
            }

            if (averageEmbeddingJson != null && !averageEmbeddingJson.isEmpty()) {
                if (averageEmbeddingJson.startsWith("[")) {
                    // It's already JSON format
                    averageEmbedding = objectMapper.readValue(averageEmbeddingJson, Double[].class);
                } else {
                    // It might be a comma-separated string
                    averageEmbedding = Arrays.stream(averageEmbeddingJson.split(","))
                        .map(String::trim)
                        .filter(s -> !s.isEmpty())
                        .map(Double::parseDouble)
                        .toArray(Double[]::new);
                }
            } else {
                averageEmbedding = new Double[0];
            }
        } catch (Exception e) {
            System.err.println("Error converting JSON to arrays: " + e.getMessage());
            e.printStackTrace();
            // Initialize with empty arrays if conversion fails
            predictionIds = new Long[0];
            averageEmbedding = new Double[0];
        }
    }

    // Convert arrays to JSON when saving to database
    @PrePersist
    @PreUpdate
    void onSave() {
        try {
            if (predictionIds != null) {
                predictionIdsJson = objectMapper.writeValueAsString(predictionIds);
            } else {
                predictionIdsJson = "[]";
            }

            if (averageEmbedding != null) {
                averageEmbeddingJson = objectMapper.writeValueAsString(averageEmbedding);
            } else {
                averageEmbeddingJson = "[]";
            }
        } catch (Exception e) {
            System.err.println("Error converting arrays to JSON: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Rest of your methods remain unchanged...

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ResponseGroup)) {
            return false;
        }
        return id != null && id.equals(((ResponseGroup) o).id);
    }

    @Override
    public int hashCode() {
        return 31;
    }

    @Override
    public String toString() {
        return "ResponseGroup{" +
            "id=" + id +
            ", question='" + question + "'" +
            ", predictionIds='" + Arrays.toString(predictionIds) + "'" +
            ", averageEmbedding='" + Arrays.toString(averageEmbedding) + "'" +
            "}";
    }

    public ResponseGroup update() {
        return update(this);
    }

    public ResponseGroup persistOrUpdate() {
        return persistOrUpdate(this);
    }

    public static ResponseGroup update(ResponseGroup responseGroup) {
        if (responseGroup == null) {
            throw new IllegalArgumentException("responseGroup can't be null");
        }
        var entity = ResponseGroup.<ResponseGroup>findById(responseGroup.id);
        if (entity != null) {
            entity.question = responseGroup.question;
            entity.predictionIds = responseGroup.predictionIds;
            entity.averageEmbedding = responseGroup.averageEmbedding;
            
            // Explicitly trigger onSave
            entity.onSave();
            
            // Force a flush to ensure changes are written
            getEntityManager().flush();
        }
        return entity;
    }

    public static ResponseGroup persistOrUpdate(ResponseGroup responseGroup) {
        if (responseGroup == null) {
            throw new IllegalArgumentException("responseGroup can't be null");
        }
        if (responseGroup.id == null) {
            // Ensure onSave is called before persisting
            responseGroup.onSave();
            persist(responseGroup);
            // Force a flush to ensure changes are written
            getEntityManager().flush();
            return responseGroup;
        } else {
            return update(responseGroup);
        }
    }

    // Other query methods remain unchanged...
    
    public static PanacheQuery<ResponseGroup> findByQuestionId(long qid) {
        return find("select rg from ResponseGroup rg where rg.question.id = ?1", qid);
    }
    
    public static long deleteByQIds(Set<Long> qids) {
        return delete("delete from ResponseGroup rg where rg.question.id in ?1", qids);
    }
    
    public static long deleteByQId(Long qid) {
        return delete("delete from ResponseGroup rg where rg.question.id = ?1", qid);
    }
    
    public static PanacheQuery<ResponseGroup> canAccess(long responseGroupId, String login) {
        return find("select rg from ResponseGroup rg join rg.question.exam.course.rgofs as u where rg.id = ?1 and u.login = ?2", responseGroupId, login);
    }
    
    public static PanacheQuery<ResponseGroup> findByQuestion(Long questionId) {
        return find("select rg from ResponseGroup rg where rg.question.id = ?1", questionId);
    }
    
    // Updated to use a Java-side filter for finding by prediction ID
    public static PanacheQuery<ResponseGroup> findByPrediction(Long predictionId) {
        List<ResponseGroup> all = listAll();
        List<ResponseGroup> filtered = new ArrayList<>();
        
        for (ResponseGroup rg : all) {
            if (rg.predictionIds != null) {
                for (Long id : rg.predictionIds) {
                    if (id != null && id.equals(predictionId)) {
                        filtered.add(rg);
                        break;
                    }
                }
            }
        }
        
        if (filtered.isEmpty()) {
            // Return an empty result if no matches
            return find("id = -1");
        } else {
            // Get the IDs of filtered groups and use them in a query
            List<Long> ids = filtered.stream().map(rg -> rg.id).collect(Collectors.toList());
            return find("id in ?1", ids);
        }
    }
}